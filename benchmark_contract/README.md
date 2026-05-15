# PriOr-Flow Benchmark Contract

Build the benchmark image from the repository root:

```bash
docker build -f benchmark_contract/Dockerfile.benchmark -t priorflow-benchmark .
```

With Podman, use:

```bash
podman build -f benchmark_contract/Dockerfile.benchmark -t priorflow-benchmark .
```

Run a scenario with NVIDIA GPU access:

```bash
docker run --rm --gpus all \
  -v priorflow-data:/data \
  -v priorflow-results:/app/benchmark_contract/results \
  priorflow-benchmark ./benchmark_contract/entrypoint.sh official_reproduction
```

Podman run equivalent on a Linux host with NVIDIA Container Toolkit/CDI configured:

```bash
podman run --rm --device nvidia.com/gpu=all \
  -v priorflow-data:/data \
  -v priorflow-results:/app/benchmark_contract/results \
  priorflow-benchmark ./benchmark_contract/entrypoint.sh official_reproduction
```

Supported scenarios:

```bash
./benchmark_contract/entrypoint.sh official_reproduction
./benchmark_contract/entrypoint.sh regional_robustness
./benchmark_contract/entrypoint.sh standardized_efficiency
```

The image keeps the PriOr-Flow project stack at PyTorch 1.12.1 with CUDA 11.3
inside the container. A newer Linux host driver/CUDA installation can still
provide the GPU through NVIDIA Container Toolkit and `--gpus all`.

Runtime downloads are cached in `/data`:

- `/data/FlowScape` for the default FlowScape dataset.
- `/data/checkpoints/priorflow/FlowScape-final.pth` for the PriOr-RAFT checkpoint.

Set `PRIORFLOW_SKIP_DOWNLOAD=1` to require pre-mounted assets instead of
downloading them.
