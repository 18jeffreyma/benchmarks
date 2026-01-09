# SWE-fficiency Evaluation

This folder contains the OpenHands inference implementation for the [SWE-fficiency benchmark](https://swefficiency.com/) ([paper](https://arxiv.org/pdf/2507.12415v1)).

SWE-fficiency is a benchmark for evaluating AI agents' ability to optimize code performance. Unlike SWE-bench which focuses on bug fixes, SWE-fficiency measures how well agents can improve the runtime of specific workloads by modifying repository source code.

## Overview

The evaluation consists of three steps:

1. **Environment setup**: Set up the evaluation environment and LLM configuration
2. **Run inference**: Generate optimization patches for each performance workload
3. **Evaluate patches**: Use the official SWE-fficiency benchmark evaluation

## Setup

### Prerequisites

- Python 3.12+
- Docker (for local workspace execution)
- uv (for package management)

### Installation

```bash
# Install dependencies
uv sync

# Set up LLM configuration
# Create a JSON file with your LLM configuration
cat > llm_config.json << EOF
{
  "model": "gpt-4o",
  "api_key": "your-api-key-here"
}
EOF
```

## Running Inference

### Basic Usage

```bash
# Run with default settings
swefficiency-infer llm_config.json

# Or using uv run
uv run swefficiency-infer llm_config.json
```

### Command Line Options

```bash
swefficiency-infer llm_config.json \
  --dataset swefficiency/swefficiency \
  --split test \
  --max-iterations 100 \
  --num-workers 1 \
  --output-dir ./eval_outputs \
  --n-limit 10 \
  --workspace docker
```

#### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `llm_config_path` | Path to JSON LLM configuration | Required |
| `--dataset` | Dataset name | `swefficiency/swefficiency` |
| `--split` | Dataset split | `test` |
| `--workspace` | Workspace type (`docker` or `remote`) | `docker` |
| `--max-iterations` | Maximum iterations per instance | `100` |
| `--num-workers` | Number of parallel workers | `1` |
| `--output-dir` | Evaluation output directory | `./eval_outputs` |
| `--n-limit` | Limit number of instances to evaluate | `0` (all) |
| `--max-attempts` | Maximum attempts for iterative mode | `3` |
| `--note` | Evaluation note for output directory | `initial` |
| `--select` | Path to file with instance IDs to select | None |
| `--prompt-path` | Path to prompt template file | Default template |
| `--enable-cpu-pinning` | Enable CPU pinning for consistent performance measurements | `False` |
| `--cpus-per-worker` | Number of CPUs per worker when CPU pinning is enabled | `4` |
| `--cleanup-image` | Delete Docker images after each instance (saves disk space) | `False` |
| `--mem-limit` | Memory limit for Docker containers | `16g` |

### SWE-fficiency Specific Features

#### CPU Pinning

For consistent performance measurements, enable CPU pinning to allocate dedicated CPU cores to each worker:

```bash
swefficiency-infer llm_config.json \
  --num-workers 4 \
  --enable-cpu-pinning \
  --cpus-per-worker 4
```

This is important for SWE-fficiency because performance measurements need to be reproducible.

#### Image Cleanup

When evaluating many instances, Docker images can consume significant disk space. Enable automatic cleanup:

```bash
swefficiency-infer llm_config.json --cleanup-image
```

This will delete each Docker image after the corresponding instance evaluation completes.

### Building Docker Images

Before running inference, you may need to build the Docker images:

```bash
# Build images for all instances in the dataset
python benchmarks/swefficiency/build_images.py \
  --dataset swefficiency/swefficiency \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal

# Skip building if images already exist
export SKIP_BUILD=1
swefficiency-infer llm_config.json
```

### Remote Workspace

For running with remote runtime:

```bash
export RUNTIME_API_KEY="your-runtime-api-key"
export RUNTIME_API_URL="https://runtime.eval.all-hands.dev"

swefficiency-infer llm_config.json --workspace remote
```

## Output

The evaluation produces output in JSONL format containing:

- `instance_id`: Unique identifier for the instance
- `test_result`: Contains the `git_patch` with the agent's changes
- `instruction`: The prompt sent to the agent
- `history`: Full conversation history
- `metrics`: Performance metrics (tokens, costs, etc.)
- `error`: Any errors encountered

Example output location:
```
eval_outputs/swefficiency__swefficiency-test/<model>/maxiter_100/<timestamp>/output.jsonl
```

## Evaluation

Once the inference output is generated, use the [official SWE-fficiency benchmark evaluation](https://github.com/swefficiency/swefficiency) to evaluate the patches.

## Dataset Structure

Each instance in the SWE-fficiency dataset contains:

- `instance_id`: Unique identifier
- `repo`: Repository name (e.g., `scikit-learn/scikit-learn`)
- `version`: Repository version
- `workload`: Python code defining the performance workload
- `test_cmd`: Command to run tests
- `rebuild_cmd`: Command to rebuild the repository
- `base_commit`: Base commit hash

## Architecture

The implementation follows the standard OpenHands benchmark architecture:

- `run_infer.py`: Main inference runner extending the `Evaluation` base class
- `build_images.py`: Docker image building utilities
- `prompts/default.j2`: Jinja2 template for agent instructions
- Uses the shared evaluation infrastructure from `benchmarks/utils/`

## References

- [SWE-fficiency Website](https://swefficiency.com/)
- [SWE-fficiency Paper](https://arxiv.org/pdf/2507.12415v1)
- [SWE-fficiency GitHub](https://github.com/swefficiency/swefficiency)
