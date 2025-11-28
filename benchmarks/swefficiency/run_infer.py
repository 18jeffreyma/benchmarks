import multiprocessing
import os
from pathlib import Path
from queue import Empty
from typing import Any, List, Optional

from jinja2 import Environment, FileSystemLoader

from benchmarks.swefficiency.binary_patch_utils import remove_binary_diffs
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.build_utils import build_image
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.image_utils import image_exists
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.version import SDK_SHORT_SHA
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import APIRemoteWorkspace, DockerWorkspace


logger = get_logger(__name__)


def get_official_docker_image(
    instance_id: str,
    docker_image_prefix="ghcr.io/swefficiency/swefficiency-images",
) -> str:
    """Get the official Docker image for a SWE-fficiency instance."""
    # SWE-fficiency uses similar image naming convention as SWE-Bench
    return f"{docker_image_prefix}:{instance_id}"


def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWE-fficiency instance ID from official image name.

    Example:
            ghcr.io/swefficiency/swefficiency-images:django-12155
            -> django-12155
    """
    return base_image.split(":")[-1]


def divide_cpus_among_workers(
    num_workers: int, num_to_skip: int = 8
) -> List[List[int]]:
    """
    Divide available CPUs among workers, skipping the first num_to_skip CPUs.

    Args:
        num_workers: Number of worker processes
        num_to_skip: Number of CPUs to skip at the beginning (default: 8)

    Returns:
        List of CPU groups, where each group is a list of CPU IDs
    """
    try:
        import psutil

        # Get all available CPUs
        cpu_count = psutil.cpu_count(logical=True)
        if cpu_count is None:
            cpu_count = multiprocessing.cpu_count()

        if num_to_skip >= cpu_count:
            logger.warning(
                f"num_to_skip ({num_to_skip}) >= cpu_count ({cpu_count}). "
                f"Using all {cpu_count} CPUs."
            )
            available_cpus = list(range(cpu_count))
        else:
            available_cpus = list(range(num_to_skip, cpu_count))

        if len(available_cpus) == 0:
            logger.warning(
                f"No CPUs available after skipping {num_to_skip}. "
                f"Using all {cpu_count} CPUs."
            )
            available_cpus = list(range(cpu_count))

        # Divide CPUs among workers
        cpus_per_worker = len(available_cpus) // num_workers
        if cpus_per_worker == 0:
            cpus_per_worker = 1

        cpu_groups = []
        for i in range(num_workers):
            start_idx = i * cpus_per_worker
            end_idx = start_idx + cpus_per_worker
            if i == num_workers - 1:
                # Last worker gets remaining CPUs
                end_idx = len(available_cpus)
            cpu_groups.append(available_cpus[start_idx:end_idx])

        logger.info(
            f"Divided {len(available_cpus)} CPUs among {num_workers} workers: "
            f"{[len(group) for group in cpu_groups]} CPUs per worker"
        )
        return cpu_groups
    except ImportError:
        logger.warning("psutil not available, CPU pinning disabled")
        return [[] for _ in range(num_workers)]


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance["repo"].split("/")[-1]
    assert metadata.details is not None

    # Set up Jinja2 environment
    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
    }
    context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction


class SWEEfficiencyEvaluation(Evaluation):
    """
    Process-based SWE-fficiency evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def __init__(self, metadata: EvalMetadata, num_workers: int = 1):
        """Initialize evaluation with CPU queue support."""
        super().__init__(metadata=metadata, num_workers=num_workers)
        # Create CPU queue for worker processes
        self._cpu_groups_queue: Any = None
        self._setup_cpu_queue()

    def _setup_cpu_queue(self) -> None:
        """Set up CPU groups queue for worker processes."""
        cpu_groups_list = divide_cpus_among_workers(self.num_workers, num_to_skip=8)
        manager = multiprocessing.Manager()
        self._cpu_groups_queue = manager.Queue()
        for cpu_group in cpu_groups_list:
            if self._cpu_groups_queue is not None:
                self._cpu_groups_queue.put(cpu_group)
        logger.info(
            f"Created CPU queue with {len(cpu_groups_list)} CPU groups "
            f"for {self.num_workers} workers"
        )

    def _get_cpu_group(self) -> Optional[List[int]]:
        """Get a CPU group from the queue (thread-safe)."""
        if self._cpu_groups_queue is None:
            return None
        try:
            return self._cpu_groups_queue.get(timeout=1)
        except Empty:
            logger.warning("CPU queue empty, no CPU pinning available")
            return None

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up SWE-fficiency evaluation data")

        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    # ---- Hook: prepare a workspace per instance ----------------------------------
    def prepare_workspace(
        self, instance: EvalInstance, cpu_group: Optional[List[int]] = None
    ) -> RemoteWorkspace:
        """
        Use DockerWorkspace by default.
        If cpu_group is provided, configure CPU pinning for the workspace.
        """
        official_docker_image = get_official_docker_image(instance.id)
        build_target = "source-minimal"
        custom_tag = extract_custom_tag(official_docker_image)
        # For non-binary targets, append target suffix
        suffix = f"-{build_target}" if build_target != "binary" else ""

        if self.metadata.workspace_type == "docker":
            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
            )
            SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
            logger.info(f"SKIP_BUILD={SKIP_BUILD}")
            if not SKIP_BUILD:
                logger.info(
                    f"Building workspace from {official_docker_image} "
                    f"for instance {instance.id}. "
                    "This may take a while...\n"
                    "You can set SKIP_BUILD=1 to skip building and use pre-built "
                    "agent-server image."
                )
                output = build_image(
                    base_image=official_docker_image,
                    target_image=EVAL_AGENT_SERVER_IMAGE,
                    custom_tag=custom_tag,
                    target=build_target,
                    push=False,
                )
                logger.info(f"Image build output: {output}")
                assert output.error is None, f"Image build failed: {output.error}"
                if agent_server_image not in output.tags:
                    raise RuntimeError(
                        f"Built image tags {output.tags} do not include expected tag "
                        f"{agent_server_image}"
                    )

            # Configure CPU pinning if cpu_group is provided
            docker_kwargs = {}
            if cpu_group is not None and len(cpu_group) > 0:
                logger.info(
                    f"Configuring Docker runtime with CPU group: {cpu_group} "
                    f"for instance {instance.id}"
                )
                # Note: DockerWorkspace may not directly support these kwargs yet,
                # but we set them up for future support
                docker_kwargs = {
                    "cpuset_cpus": ",".join(map(str, cpu_group)),
                    "nano_cpus": int(1e9 * len(cpu_group)),  # 1 CPU = 1e9 nano CPUs
                    "mem_limit": "16g",
                }
                logger.debug(f"Docker kwargs: {docker_kwargs}")

            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                **docker_kwargs,
            )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            sdk_short_sha = os.getenv("SDK_SHORT_SHA", SDK_SHORT_SHA)
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{sdk_short_sha}-{custom_tag}{suffix}"
            )
            if not image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry, "
                    "make sure to build, push it, and make it public accessible before using remote workspace."
                )
            logger.info(
                f"Using remote workspace with image {agent_server_image} (sdk sha: {sdk_short_sha})"
            )
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                target_type="source" if "source" in build_target else "binary",
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        for cmd in self.metadata.env_setup_commands or []:
            res = workspace.execute_command(cmd)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Failed to run env setup command '{cmd}': {res.stderr}"
                )
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
        return workspace

    # ---- Hook: evaluate one instance ---------------------------------------------
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Create conversation, run agent, collect history and git patch.
        Do not write files here; just return EvalOutput.
        """
        tools = get_default_tools(
            # Disable browser tools in CLI mode
            enable_browser=False,
        )
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        assert isinstance(workspace, RemoteWorkspace)

        def _log_event(ev):  # keep it simple
            logger.debug("Event: %s", ev)

        repo_path = f"/workspace/{instance.data['repo'].split('/')[-1]}/"
        instance.data["repo_path"] = repo_path

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[_log_event],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        logger.info("repo_path: %s", repo_path)
        cp_testebed_repo = workspace.execute_command(
            (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
        )
        assert cp_testebed_repo.exit_code == 0, (
            f"cp_testebed_repo failed: {cp_testebed_repo.stderr}"
        )

        # git reset
        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)
        conversation.run()

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # git commit
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit -m 'patch'"
        )

        # Get git patch
        base_commit = instance.data["base_commit"]
        git_patch_result = workspace.execute_command(
            (f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD")
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        git_patch = git_patch_result.stdout

        # Remove binary diffs from patch
        git_patch = remove_binary_diffs(git_patch)

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
            test_result={
                "git_patch": git_patch,
            },
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out

    # Override _process_one_mp to get CPU group from queue
    def _process_one_mp(
        self, instance: EvalInstance
    ) -> tuple[EvalInstance, EvalOutput]:
        """Execute one instance in a child process with CPU pinning support.

        Gets CPU group from queue and passes it to prepare_workspace.
        """
        from benchmarks.utils.evaluation import (
            redirect_stdout_stderr,
            reset_logger_for_multiprocessing,
        )

        # Set up instance-specific logging
        log_dir = os.path.join(self.metadata.eval_output_dir, "logs")
        reset_logger_for_multiprocessing(log_dir, instance.id)

        # Get log file path for stdout/stderr redirection
        log_file = os.path.join(log_dir, f"instance_{instance.id}.output.log")

        # Redirect stdout/stderr to capture all output
        with redirect_stdout_stderr(log_file):
            logger.info("[child] start id=%s", instance.id)

            # Get CPU group from queue (if available)
            cpu_group = self._get_cpu_group()
            if cpu_group:
                logger.info(
                    f"[child] Using CPU group {cpu_group} for instance {instance.id}"
                )

            retry_count = 0
            last_error = None
            max_retries = self.metadata.max_retries

            while retry_count <= max_retries:
                workspace = None
                try:
                    # Pass cpu_group to prepare_workspace
                    workspace = self.prepare_workspace(instance, cpu_group=cpu_group)
                    out = self.evaluate_instance(instance, workspace)

                    # Capture conversation archive after successful evaluation
                    self._capture_conversation_archive(workspace, instance)

                    logger.info("[child] done id=%s", instance.id)
                    return instance, out
                except Exception as e:
                    last_error = e
                    retry_count += 1

                    if retry_count <= max_retries:
                        logger.warning(
                            f"[child] Instance {instance.id} failed "
                            f"(attempt {retry_count}/{max_retries}): "
                            f"{str(e)[:50]}"
                        )
                    else:
                        logger.error(
                            f"[child] Instance {instance.id} failed after "
                            f"{max_retries} retries. Last error: {str(e)[:50]}",
                            exc_info=True,
                        )
                        # Create error output for final failure
                        error_output = self._create_error_output(
                            instance, last_error, max_retries
                        )
                        return instance, error_output
                finally:
                    # Ensure workspace cleanup happens regardless of success or failure
                    if workspace is not None:
                        try:
                            # Use the context manager protocol for cleanup
                            workspace.__exit__(None, None, None)
                            logger.debug(
                                "[child] cleaned up workspace for id=%s", instance.id
                            )
                        except Exception as cleanup_error:
                            logger.warning(
                                f"[child] Failed to cleanup workspace for {instance.id}: "
                                f"{str(cleanup_error)[:50]}"
                            )

            # This should never be reached, but added for type safety
            error_output = self._create_error_output(
                instance, Exception("Unexpected error: no attempts made"), max_retries
            )
            return instance, error_output


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
    default_prompt_path = prompt_dir / "default.j2"
    assert default_prompt_path.exists(), (
        f"Default prompt {default_prompt_path} not found"
    )

    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        choices=choices,
        help="Path to prompt template file",
    )
    args = parser.parse_args()

    # Validate max_attempts
    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    # Create critic instance from parsed arguments
    critic = create_critic(args)
    logger.info(f"Using critic: {type(critic).__name__}")

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
    )

    # Run orchestrator with a simple JSONL writer
    evaluator = SWEEfficiencyEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
