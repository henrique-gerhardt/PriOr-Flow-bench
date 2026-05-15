from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import yaml


CONTRACT_ROOT = Path(os.environ.get("PRIORFLOW_CONTRACT_ROOT", "/app/benchmark_contract"))
RESULTS_DIR = Path(os.environ.get("PRIORFLOW_RESULTS_DIR", str(CONTRACT_ROOT / "results")))


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def checkpoint_size_mb(path: str) -> Optional[float]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return None
    return checkpoint.stat().st_size / (1024.0 * 1024.0)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_model(method_root: Path, checkpoint: str, mixed_precision: bool) -> torch.nn.Module:
    sys.path.insert(0, str(method_root))
    sys.path.insert(0, str(method_root / "core"))
    from core.prior_raft import PriOr_RAFT

    model_args = SimpleNamespace(mixed_precision=mixed_precision, dropout=0.0)
    model = torch.nn.DataParallel(PriOr_RAFT(model_args), device_ids=[0])
    state = torch.load(checkpoint, map_location="cuda")
    model.load_state_dict(state, strict=True)
    model.cuda()
    model.eval()
    return model


def measure_latency(model: torch.nn.Module, height: int, width: int, warmup: int, runs: int, iters: int) -> Dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for standardized_efficiency.")
    device = torch.device("cuda")
    image1 = torch.rand((1, 3, height, width), device=device, dtype=torch.float32) * 255.0
    image2 = torch.rand((1, 3, height, width), device=device, dtype=torch.float32) * 255.0

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(image1, image2, iters=iters, test_mode=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(image1, image2, iters=iters, test_mode=True)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(timings) if timings else None
    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": statistics.median(timings) if timings else None,
        "latency_p95_ms": percentile(timings, 95.0),
        "max_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "fps": 1000.0 / mean_ms if mean_ms and mean_ms > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["standardized_efficiency"], required=True)
    args = parser.parse_args()

    runtime = load_yaml(CONTRACT_ROOT / "config" / "runtime.yaml")
    experiment = load_yaml(CONTRACT_ROOT / "config" / "experiment.yaml")
    method_root = Path(experiment.get("method_root", "/app/PriOr-RAFT"))
    checkpoint = experiment["checkpoint"]
    height = int(runtime.get("standardized_efficiency", {}).get("input_height", 512))
    width = int(runtime.get("standardized_efficiency", {}).get("input_width", 1024))
    warmup = int(os.environ.get("PRIORFLOW_WARMUP_RUNS", runtime.get("warmup_runs", 10)))
    runs = int(os.environ.get("PRIORFLOW_MEASURED_RUNS", runtime.get("measured_runs", 50)))
    iters = int(os.environ.get("PRIORFLOW_INFERENCE_ITERS", runtime.get("inference_iters", 12)))
    mixed_precision = runtime.get("precision") in {"fp16", "mixed", "mixed_precision"}

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for standardized_efficiency.")
    model = load_model(method_root, checkpoint, mixed_precision=mixed_precision)
    metrics = measure_latency(model, height, width, warmup, runs, iters)
    metrics.update(
        {
            "parameters": count_parameters(model.module if hasattr(model, "module") else model),
            "checkpoint_size_mb": checkpoint_size_mb(checkpoint),
            "flops_g": None,
            "flops_note": "Not reported: PriOr-RAFT uses dynamic spherical resampling and all-pairs correlation paths that are not reliably counted by fvcore in this wrapper.",
            "scenario": args.scenario,
            "warmup_runs": warmup,
            "measured_runs": runs,
            "input_height": height,
            "input_width": width,
        }
    )
    write_json(RESULTS_DIR / "efficiency_metrics.json", metrics)


if __name__ == "__main__":
    main()
