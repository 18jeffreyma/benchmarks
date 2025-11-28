# SWE-fficiency Benchmark Evaluation

This directory contains the implementation for running SWE-fficiency evaluation using OpenHands agents.

## Overview

SWE-fficiency is a benchmark for evaluating AI agents on real-world software engineering tasks derived from GitHub issues. Similar to SWE-Bench, it tests an agent's ability to understand problem statements, navigate codebases, and generate patches that resolve issues. SWE-fficiency includes additional utilities for handling binary files in patches.

## Dataset

- **Source**: Similar to SWE-Bench datasets
- **Datasets**: 
  - `princeton-nlp/SWE-bench` - Full dataset
  - `princeton-nlp/SWE-bench_Lite` - Smaller curated subset
  - `princeton-nlp/SWE-bench_Verified` - Verified instances
- **Splits**: `test`, `dev`

## Features

- **Binary File Handling**: Automatic removal of binary file diffs from generated patches
- **Docker Workspace Support**: Local evaluation using Docker containers
- **Remote Workspace Support**: Scalable cloud-based evaluation with parallelization
- **Patch Post-processing**: Clean patches by removing binary files and unnecessary changes

## Usage

### Docker Workspace (Local Evaluation)

#### Step 1: Build Docker Images

Before running inference, you need to build Docker images for the SWE-fficiency instances. Each instance requires a specific environment setup based on the repository and issue.

```bash
# Build images using SWE-Bench build script (shared infrastructure)
uv run python -m benchmarks.swe_bench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/agent-server \
  --target source-minimal
```

#### Step 2: Run Inference

Run evaluation using the built Docker images:

```bash
uv run swefficiency-infer path/to/llm_config.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --max-iterations 100 \
    --workspace docker
```

**Selecting specific instances:**

You can run evaluation on a specific subset by creating a text file with instance IDs:

```bash
# Create instances.txt with one instance ID per line
echo "django__django-11333" > instances.txt
echo "astropy__astropy-12345" >> instances.txt

# Run with selection
uv run swefficiency-infer path/to/llm_config.json \
    --select instances.txt \
    --workspace docker
```

### Remote Workspace (Scalable Cloud Evaluation)

Remote workspace enables running evaluations at scale by using a cloud-based runtime API to provision containers. This is ideal for large-scale benchmark runs with high parallelization.

#### Step 1: Pre-build and Push Images

Images must be pre-built and pushed to a **public** container registry before running remote evaluations.

```bash
uv run python -m benchmarks.swe_bench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --push \
  --max-workers 32
```

**Important Notes:**
- Images must be **publicly accessible** for the remote runtime to pull them
- The SDK SHA is automatically detected from the `vendor/software-agent-sdk` submodule
- Each SWE-fficiency instance gets its own unique image tag based on the repository and issue

#### Step 2: Set Up Environment Variables

```bash
# Required: Your runtime API key
export RUNTIME_API_KEY="your-runtime-api-key-here"

# Optional: Override default runtime API URL
export RUNTIME_API_URL="https://runtime.eval.all-hands.dev"

# Optional: Override SDK SHA for image selection
# (defaults to auto-detected from vendor/software-agent-sdk submodule)
export SDK_SHORT_SHA="abc1234"
```

#### Step 3: Run Inference with Remote Workspace

Run evaluation using the remote workspace with high parallelization:

```bash
uv run swefficiency-infer .llm_config/sonnet-4-5.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --workspace remote \
    --num-workers 32 \
    --max-iterations 500 \
    --n-limit 200
```

**Command Options Explained:**
- `--workspace remote`: Use remote runtime instead of local Docker
- `--num-workers 32`: Run 32 instances in parallel (adjust based on your quota)
- `--max-iterations 500`: Maximum steps per instance (higher for complex tasks)
- `--n-limit 200`: Limit to first 200 instances (optional, for testing)

### Using Shell Scripts

For convenience, you can use the provided shell scripts:

```bash
# Basic usage
./benchmarks/swefficiency/scripts/run_infer.sh \
    .llm_config/example.json \
    princeton-nlp/SWE-bench_Verified \
    test \
    100 \
    docker
```

## Binary File Handling

SWE-fficiency automatically removes binary file diffs from generated patches to ensure clean, text-only patches. This is handled by the `binary_patch_utils.py` module which:

- Detects binary file changes in git patches
- Removes binary file diffs from the final patch output
- Provides utilities for removing binary files from git staging

## Evaluation

After running inference (with either workspace type), the generated patches are automatically cleaned of binary files. The output format is compatible with SWE-Bench evaluation tools.

**Basic evaluation:**

```bash
# Use SWE-Bench evaluation script (shared infrastructure)
uv run swebench-eval output.jsonl
```

## Differences from SWE-Bench

SWE-fficiency extends SWE-Bench with:

1. **Binary Patch Handling**: Automatic removal of binary file diffs
2. **Enhanced Patch Cleaning**: Additional utilities for patch post-processing
3. **Setup Scripts**: Convenience scripts for environment setup and evaluation

## References

- [SWE-Bench Paper](https://arxiv.org/abs/2310.06770)
- [SWE-Bench GitHub](https://github.com/princeton-nlp/SWE-bench)
- [SWE-Bench Leaderboard](https://www.swebench.com/)

