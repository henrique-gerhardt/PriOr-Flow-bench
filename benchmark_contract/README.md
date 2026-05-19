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

- `/data/MPFDataset` for the default MPFDataset/EFT reproduction dataset.
- `/data/checkpoints/priorflow/EFT-final.pth` for the default PriOr-RAFT checkpoint.

MPFDataset is now the default because FlowScape is often inaccessible through
the public Google Drive link. Download the MPFDataset archive from the official
project link listed at https://github.com/HenryLee0314/ECCV2022-MPF-net and
place it in the data volume as `/data/downloads/MPFDataset.zip`. The next run
will extract it to `/data/MPFDataset`.

You can also point to a mounted archive explicitly:

```bash
podman run --rm --device nvidia.com/gpu=all \
  -e PRIORFLOW_MPFDATASET_ARCHIVE=/data/downloads/<downloaded-archive> \
  -v priorflow-data:/data \
  -v priorflow-results:/app/benchmark_contract/results \
  priorflow-benchmark ./benchmark_contract/entrypoint.sh official_reproduction
```

If you later want to use FlowScape, download the archive from one of the
official PanoFlow mirrors listed at
https://github.com/MasterHow/PanoFlow#flowscape-flow360-dataset and place it in
the data volume as `/data/downloads/FlowScape.zip`. The next run will extract it
to `/data/FlowScape`.

FlowScape can also be specified explicitly:

```bash
podman run --rm --device nvidia.com/gpu=all \
  -e PRIORFLOW_FLOWSCAPE_ARCHIVE=/data/downloads/<downloaded-archive> \
  -v priorflow-data:/data \
  -v priorflow-results:/app/benchmark_contract/results \
  priorflow-benchmark ./benchmark_contract/entrypoint.sh official_reproduction
```

Set `PRIORFLOW_SKIP_DOWNLOAD=1` to require pre-mounted assets instead of
downloading them.
