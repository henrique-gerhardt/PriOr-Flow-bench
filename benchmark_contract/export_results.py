from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional

import yaml

try:
    import torch
except Exception:
    torch = None


CONTRACT_ROOT = Path(os.environ.get("PRIORFLOW_CONTRACT_ROOT", "/app/benchmark_contract"))
RESULTS_DIR = Path(os.environ.get("PRIORFLOW_RESULTS_DIR", str(CONTRACT_ROOT / "results")))


QUALITY_DEFAULTS = {
    "epe_global": None,
    "epe_polar": None,
    "epe_equatorial": None,
    "epe_by_latitude": None,
    "valid_pixels_ratio": None,
}

EFFICIENCY_DEFAULTS = {
    "parameters": None,
    "checkpoint_size_mb": None,
    "flops_g": None,
    "latency_mean_ms": None,
    "latency_median_ms": None,
    "latency_p95_ms": None,
    "max_gpu_memory_mb": None,
    "fps": None,
}


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def try_git_commit(root: str = "/app") -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def checkpoint_size_mb(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    checkpoint = Path(path)
    if not checkpoint.exists():
        return None
    return checkpoint.stat().st_size / (1024.0 * 1024.0)


def environment_payload() -> Dict:
    if torch is None:
        return {
            "framework": "pytorch",
            "framework_version": None,
            "python_version": platform.python_version(),
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "device_count": 0,
            "platform": platform.platform(),
            "note": "PyTorch is not importable in this Python environment.",
        }
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {
        "framework": "pytorch",
        "framework_version": getattr(torch, "__version__", None),
        "python_version": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": gpu_name,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "platform": platform.platform(),
    }


def merge_defaults(path: Path, defaults: Dict) -> Dict:
    payload = dict(defaults)
    payload.update(read_json(path))
    write_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["preflight", "finalize"])
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()

    manifest = load_yaml(CONTRACT_ROOT / "manifest.yaml")
    runtime = load_yaml(CONTRACT_ROOT / "config" / "runtime.yaml")
    experiment = load_yaml(CONTRACT_ROOT / "config" / "experiment.yaml")

    metadata = read_json(RESULTS_DIR / "metadata.json")
    metadata.update(
        {
            "method_name": manifest.get("method_name"),
            "method_family": manifest.get("method_family"),
            "paper_year": manifest.get("paper_year"),
            "dataset": experiment.get("dataset"),
            "scene": experiment.get("scene"),
            "scenario": args.scenario,
            "checkpoint": experiment.get("checkpoint"),
            "framework": manifest.get("framework"),
            "commit": try_git_commit("/app"),
            "exit_code": args.exit_code,
        }
    )

    efficiency_cfg = runtime.get("standardized_efficiency", {})
    run_config = {
        "scenario": args.scenario,
        "batch_size": runtime.get("batch_size"),
        "precision": runtime.get("precision"),
        "input_height": efficiency_cfg.get("input_height"),
        "input_width": efficiency_cfg.get("input_width"),
        "warmup_runs": runtime.get("warmup_runs"),
        "measured_runs": runtime.get("measured_runs"),
        "dataset": experiment.get("dataset"),
        "scene": experiment.get("scene"),
    }

    write_json(RESULTS_DIR / "metadata.json", metadata)
    write_json(RESULTS_DIR / "run_config.json", run_config)
    write_json(RESULTS_DIR / "environment.json", environment_payload())
    merge_defaults(RESULTS_DIR / "quality_metrics.json", QUALITY_DEFAULTS)
    efficiency = merge_defaults(RESULTS_DIR / "efficiency_metrics.json", EFFICIENCY_DEFAULTS)
    if efficiency.get("checkpoint_size_mb") is None:
        efficiency["checkpoint_size_mb"] = checkpoint_size_mb(experiment.get("checkpoint"))
        write_json(RESULTS_DIR / "efficiency_metrics.json", efficiency)


if __name__ == "__main__":
    main()
