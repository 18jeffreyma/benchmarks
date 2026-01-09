import multiprocessing
import os
from pathlib import Path
from typing import Any, List

from jinja2 import Environment, FileSystemLoader
from pydantic import Field

from benchmarks.swefficiency.build_images import (
    extract_custom_tag,
    get_official_docker_image,
)
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


# ============================================================================
# CPU Pinning Support for SWE-fficiency
# ============================================================================

# Global CPU group queue for distributing CPU groups to workers
_cpu_groups_queue: Any = None


def divide_cpus_among_workers(
    num_workers: int,
    num_cpus_per_worker: int = 4,
    num_to_skip: int = 0,
) -> list[list[int]]:
    """Divide available CPUs among workers for CPU pinning.

    Args:
        num_workers: Number of worker processes
        num_cpus_per_worker: CPUs to allocate per worker
        num_to_skip: Number of initial CPUs to skip (for OS overhead)

    Returns:
        List of CPU groups, where each group is a list of CPU indices
    """
    try:
        current_cpus = list(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity not available on all platforms
        current_cpus = list(range(multiprocessing.cpu_count()))

    num_cpus = len(current_cpus)
    if num_workers <= 0:
        raise ValueError("Number of workers must be greater than 0")

    # Check that num workers and num_cpus_per_worker fit into available CPUs
    total_cpus_needed = num_workers * num_cpus_per_worker + num_to_skip
    if total_cpus_needed > num_cpus:
        logger.warning(
            f"Not enough CPUs for pinning. Requested {total_cpus_needed} "
            f"CPUs (num_workers={num_workers}, num_cpus_per_worker={num_cpus_per_worker}, "
            f"num_to_skip={num_to_skip}), but only {num_cpus} CPUs are available. "
            "CPU pinning will be disabled."
        )
        return []

    # Divide this into groups, skipping the first `num_to_skip` CPUs.
    available_cpus = current_cpus[num_to_skip:]
    cpu_groups = [
        available_cpus[i * num_cpus_per_worker : (i + 1) * num_cpus_per_worker]
        for i in range(num_workers)
    ]
    logger.info(
        f"Divided {num_cpus} CPUs into {num_workers} groups, "
        f"each with {num_cpus_per_worker} CPUs: {cpu_groups}"
    )
    return cpu_groups


def get_cpu_group_for_worker() -> list[int] | None:
    """Get a CPU group from the global queue for the current worker.

    Returns:
        List of CPU indices for this worker, or None if CPU pinning is disabled
    """
    import queue

    global _cpu_groups_queue
    if _cpu_groups_queue is not None:
        try:
            cpu_group = _cpu_groups_queue.get_nowait()
            logger.info(f"Worker acquired CPU group: {cpu_group}")
            return cpu_group
        except queue.Empty:
            logger.debug("No CPU groups available in queue")
            return None
        except Exception as e:
            logger.warning(f"Failed to acquire CPU group: {e}")
            return None
    return None


def release_cpu_group_for_worker(cpu_group: list[int] | None) -> None:
    """Release a CPU group back to the global queue.

    Args:
        cpu_group: The CPU group to release
    """
    import queue

    global _cpu_groups_queue
    if _cpu_groups_queue is not None and cpu_group is not None:
        try:
            _cpu_groups_queue.put_nowait(cpu_group)
            logger.info(f"Worker released CPU group: {cpu_group}")
        except queue.Full:
            logger.warning(f"Queue is full, could not release CPU group: {cpu_group}")
        except Exception as e:
            logger.warning(f"Failed to release CPU group: {e}")


def init_cpu_pinning(num_workers: int, num_cpus_per_worker: int = 4) -> None:
    """Initialize CPU pinning for the evaluation.

    Args:
        num_workers: Number of worker processes
        num_cpus_per_worker: CPUs to allocate per worker
    """
    global _cpu_groups_queue
    cpu_groups = divide_cpus_among_workers(num_workers, num_cpus_per_worker)
    if cpu_groups:
        _cpu_groups_queue = multiprocessing.Manager().Queue()
        for group in cpu_groups:
            _cpu_groups_queue.put(group)


# ============================================================================
# CPU-Pinned Docker Workspace for SWE-fficiency
# ============================================================================


class CPUPinnedDockerWorkspace(DockerWorkspace):
    """DockerWorkspace with CPU pinning and resource limits.

    This workspace extends DockerWorkspace to add CPU pinning support
    for SWE-fficiency benchmark, which requires consistent CPU allocation
    for performance measurements.
    """

    # CPU pinning configuration - using Pydantic Field for proper typing
    cpu_group: list[int] | None = Field(
        default=None,
        description="List of CPU indices to pin the container to. If None, no CPU pinning.",
    )
    mem_limit: str = Field(
        default="16g",
        description="Memory limit for the Docker container (e.g., '16g', '8g').",
    )

    def _start_container(self, image: str, context) -> None:
        """Override to add CPU pinning flags to docker run command."""
        import threading
        import uuid

        from openhands.sdk.utils.command import execute_command

        # Store the image name for cleanup
        self._image_name = image

        # Determine port
        if self.host_port is None:
            from openhands.workspace.docker.workspace import find_available_tcp_port

            self.host_port = find_available_tcp_port()
        else:
            self.host_port = int(self.host_port)

        from openhands.workspace.docker.workspace import check_port_available

        if not check_port_available(self.host_port):
            raise RuntimeError(f"Port {self.host_port} is not available")

        if self.extra_ports:
            if not check_port_available(self.host_port + 1):
                raise RuntimeError(
                    f"Port {self.host_port + 1} is not available for VSCode"
                )
            if not check_port_available(self.host_port + 2):
                raise RuntimeError(
                    f"Port {self.host_port + 2} is not available for VNC"
                )

        # Ensure docker is available
        docker_ver = execute_command(["docker", "version"]).returncode
        if docker_ver != 0:
            raise RuntimeError(
                "Docker is not available. Please install and start "
                "Docker Desktop/daemon."
            )

        # Prepare Docker run flags
        flags: list[str] = []
        for key in self.forward_env:
            if key in os.environ:
                flags += ["-e", f"{key}={os.environ[key]}"]

        if self.mount_dir:
            mount_path = "/workspace"
            flags += ["-v", f"{self.mount_dir}:{mount_path}"]
            logger.info(
                "Mounting host dir %s to container path %s",
                self.mount_dir,
                mount_path,
            )

        ports = ["-p", f"{self.host_port}:8000"]
        if self.extra_ports:
            ports += [
                "-p",
                f"{self.host_port + 1}:8001",  # VSCode
                "-p",
                f"{self.host_port + 2}:8002",  # Desktop VNC
            ]
        flags += ports

        # Add GPU support if enabled
        if self.enable_gpu:
            flags += ["--gpus", "all"]

        # Add CPU pinning if specified
        if self.cpu_group:
            cpuset = ",".join(map(str, self.cpu_group))
            flags += [
                "--cpuset-cpus",
                cpuset,
                "--cpus",
                str(len(self.cpu_group)),
            ]
            logger.info(
                f"CPU pinning enabled: cpuset={cpuset}, cpus={len(self.cpu_group)}"
            )

        # Add memory limit
        if self.mem_limit:
            flags += ["--memory", self.mem_limit]
            logger.info(f"Memory limit set to: {self.mem_limit}")

        # Run container
        run_cmd = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            "--rm",
            "--name",
            f"agent-server-{uuid.uuid4()}",
            *flags,
            image,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        proc = execute_command(run_cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to run docker container: {proc.stderr}")

        self._container_id = proc.stdout.strip()
        logger.info("Started container: %s", self._container_id)

        # Optionally stream logs in background
        if self.detach_logs:
            self._logs_thread = threading.Thread(
                target=self._stream_docker_logs, daemon=True
            )
            self._logs_thread.start()

        # Set host for RemoteWorkspace to use
        object.__setattr__(self, "host", f"http://localhost:{self.host_port}")
        object.__setattr__(self, "api_key", None)

        # Wait for container to be healthy
        self._wait_for_health()
        logger.info("Docker workspace is ready at %s", self.host)

        # Now initialize the parent RemoteWorkspace with the container URL
        from openhands.sdk.workspace import RemoteWorkspace

        RemoteWorkspace.model_post_init(self, context)

    def cleanup(self) -> None:
        """Stop and remove the Docker container, then release CPU group."""
        # Release CPU group back to the queue before parent cleanup
        if self.cpu_group:
            release_cpu_group_for_worker(self.cpu_group)

        # Call parent cleanup
        super().cleanup()


# ============================================================================
# Instruction and Workspace Helpers
# ============================================================================


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = _get_workspace_dir_name(instance)
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

    # Render the instruction
    instruction = template.render(context)
    return instruction


def _get_workspace_dir_name(instance: dict) -> str:
    """Get workspace directory name for an instance.

    For SWE-fficiency, the format is repo__version.
    Uses 'default' as version if not specified.
    """
    repo = instance["repo"]
    version = instance.get("version") or "default"
    return f"{repo}__{version}".replace("/", "__")


class SWEfficiencyEvaluation(Evaluation):
    """
    Process-based SWE-fficiency evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    SWE-fficiency is a benchmark for evaluating AI agents' ability to
    optimize code performance. Unlike SWE-bench which focuses on bug fixes,
    SWE-fficiency measures how well agents can improve the runtime of
    specific workloads.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

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

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """
        Use DockerWorkspace by default.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
                           Higher values allocate more CPU/memory resources.
                           Used by APIRemoteWorkspace for remote runtime allocation.
        """
        official_docker_image = get_official_docker_image(instance.id)
        build_target = "source-minimal"
        custom_tag = extract_custom_tag(official_docker_image)
        # For non-binary targets, append target suffix
        suffix = f"-{build_target}" if build_target != "binary" else ""
        base_agent_image = (
            f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
        )
        agent_server_image = base_agent_image

        if self.metadata.workspace_type == "docker":
            SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
            logger.info(f"SKIP_BUILD={SKIP_BUILD}")
            if not SKIP_BUILD:
                logger.info(
                    f"Building workspace from {official_docker_image} "
                    f"for instance {instance.id}. "
                    "This may take a while...\n"
                    "You can run benchmarks/swefficiency/build_images.py and set "
                    "SKIP_BUILD=1 to skip building and use pre-built "
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
                if base_agent_image not in output.tags:
                    raise RuntimeError(
                        f"Built image tags {output.tags} do not include expected tag "
                        f"{base_agent_image}"
                    )

            # Check if CPU pinning is enabled via metadata.details
            enable_cpu_pinning = self.metadata.details.get("enable_cpu_pinning", False)
            cleanup_image = self.metadata.details.get("cleanup_image", False)

            if enable_cpu_pinning:
                # Get CPU group for this worker
                cpu_group = get_cpu_group_for_worker()
                workspace = CPUPinnedDockerWorkspace(
                    server_image=agent_server_image,
                    working_dir="/workspace",
                    forward_env=forward_env or [],
                    cleanup_image=cleanup_image,
                    cpu_group=cpu_group,
                    mem_limit=self.metadata.details.get("mem_limit", "16g"),
                )
            else:
                workspace = DockerWorkspace(
                    server_image=agent_server_image,
                    working_dir="/workspace",
                    forward_env=forward_env or [],
                    cleanup_image=cleanup_image,
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
                f"Using remote workspace with image {agent_server_image} "
                f"(sdk sha: {sdk_short_sha}, resource_factor: {resource_factor})"
            )
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                target_type="source" if "source" in build_target else "binary",
                forward_env=forward_env or [],
                resource_factor=resource_factor,
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

        workspace_dir_name = _get_workspace_dir_name(instance.data)
        repo_path = f"/workspace/{workspace_dir_name}/"
        instance.data["repo_path"] = repo_path

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[_log_event],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        logger.info("repo_path: %s", repo_path)
        # Copy testbed to workspace
        cp_testbed_repo = workspace.execute_command(
            f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}"
        )
        assert cp_testbed_repo.exit_code == 0, (
            f"cp_testbed_repo failed: {cp_testbed_repo.stderr}"
        )

        # git reset
        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        # Configure git
        workspace.execute_command(
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation'"
        )

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)
        conversation.run()

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # Check if there are any changes to commit
        status_result = workspace.execute_command(
            f"cd {repo_path} ; git status --porcelain"
        )

        git_patch = ""
        base_commit = instance.data.get("base_commit")

        if status_result.exit_code == 0 and status_result.stdout.strip():
            # There are changes to commit
            commit_result = workspace.execute_command(
                f"cd {repo_path} ; git commit -m 'patch'"
            )
            if commit_result.exit_code != 0:
                logger.warning(f"git commit failed: {commit_result.stderr}")
            else:
                # Get git patch
                if base_commit:
                    # Use the base_commit from instance data
                    git_patch_result = workspace.execute_command(
                        f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD"
                    )
                else:
                    # Fall back to diff against the previous commit
                    git_patch_result = workspace.execute_command(
                        f"cd {repo_path} ; git --no-pager diff --no-color HEAD~1 HEAD"
                    )

                if git_patch_result.exit_code == 0:
                    git_patch = git_patch_result.stdout
                else:
                    logger.warning(f"git diff failed: {git_patch_result.stderr}")
                    # Try to get the last commit as a patch
                    git_patch_result = workspace.execute_command(
                        f"cd {repo_path} ; git --no-pager show --no-color HEAD"
                    )
                    if git_patch_result.exit_code == 0:
                        git_patch = git_patch_result.stdout
                    else:
                        logger.warning("git show HEAD also failed, using empty patch")
        else:
            logger.warning("No changes detected, using empty patch")

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
    # SWE-fficiency specific arguments
    parser.add_argument(
        "--enable-cpu-pinning",
        action="store_true",
        default=False,
        help="Enable CPU pinning for Docker workspaces (for consistent performance measurements)",
    )
    parser.add_argument(
        "--cpus-per-worker",
        type=int,
        default=4,
        help="Number of CPUs to allocate per worker when CPU pinning is enabled (default: 4)",
    )
    parser.add_argument(
        "--cleanup-image",
        action="store_true",
        default=False,
        help="Delete Docker images after each instance evaluation (saves disk space)",
    )
    parser.add_argument(
        "--mem-limit",
        type=str,
        default="16g",
        help="Memory limit for Docker containers (default: 16g)",
    )
    # Set defaults for SWE-fficiency
    parser.set_defaults(dataset="swefficiency/swefficiency", split="test")
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

    # Initialize CPU pinning if enabled (for Docker workspace only)
    if args.enable_cpu_pinning and args.workspace == "docker":
        logger.info(
            f"Initializing CPU pinning with {args.num_workers} workers, "
            f"{args.cpus_per_worker} CPUs per worker"
        )
        init_cpu_pinning(args.num_workers, args.cpus_per_worker)

    # Store SWE-fficiency specific options in details
    swefficiency_details = {
        "enable_cpu_pinning": args.enable_cpu_pinning,
        "cleanup_image": args.cleanup_image,
        "mem_limit": args.mem_limit,
        "cpus_per_worker": args.cpus_per_worker,
    }

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details=swefficiency_details,
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
    evaluator = SWEfficiencyEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
