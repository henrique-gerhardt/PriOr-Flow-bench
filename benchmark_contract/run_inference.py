from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


CONTRACT_ROOT = Path(os.environ.get("PRIORFLOW_CONTRACT_ROOT", "/app/benchmark_contract"))
RESULTS_DIR = Path(os.environ.get("PRIORFLOW_RESULTS_DIR", str(CONTRACT_ROOT / "results")))
OUTPUTS_DIR = Path(os.environ.get("PRIORFLOW_OUTPUTS_DIR", str(CONTRACT_ROOT / "outputs")))


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_official(method_root: Path, script: str, args: List[str], env: Dict[str, str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, script] + args
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(method_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    return proc


def parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_official_metrics(output: str) -> Dict:
    pattern = re.compile(r"Validation \((?P<label>[^)]+)\) EPE:\s*(?P<epe>[-+0-9.eE]+),\s*SEPE:\s*(?P<sepe>[-+0-9.eE]+)")
    match = pattern.search(output)
    epe = parse_float(match.group("epe")) if match else None
    sepe = parse_float(match.group("sepe")) if match else None
    return {
        "epe_global": epe,
        "epe_polar": None,
        "epe_equatorial": None,
        "epe_by_latitude": None,
        "valid_pixels_ratio": 1.0 if epe is not None else None,
        "sepe_global": sepe,
    }


def parse_regional_metrics(output: str) -> Dict:
    pattern = re.compile(
        r"^\s*(?P<region>All|Equator|Poles|Center)-(?P<label>[^:]+):\s*epe\s*(?P<epe>[-+0-9.eE]+),\s*sd\s*(?P<sd>[-+0-9.eE]+)",
        re.MULTILINE,
    )
    regions = {match.group("region"): parse_float(match.group("epe")) for match in pattern.finditer(output)}
    global_epe = regions.get("All")
    return {
        "epe_global": global_epe,
        "epe_polar": regions.get("Poles"),
        "epe_equatorial": regions.get("Equator"),
        "epe_by_latitude": {
            "all": global_epe,
            "equator": regions.get("Equator"),
            "poles": regions.get("Poles"),
            "center": regions.get("Center"),
        } if regions else None,
        "valid_pixels_ratio": 1.0 if regions else None,
    }


def update_metadata(manifest: Dict, experiment: Dict, scenario: str, command: List[str], exit_code: int) -> None:
    metadata = read_json(RESULTS_DIR / "metadata.json")
    metadata.update(
        {
            "method_name": manifest.get("method_name"),
            "method_family": manifest.get("method_family"),
            "paper_year": manifest.get("paper_year"),
            "dataset": experiment.get("dataset"),
            "scene": experiment.get("scene"),
            "scenario": scenario,
            "checkpoint": experiment.get("checkpoint"),
            "framework": manifest.get("framework"),
            "official_command": command,
            "official_command_exit_code": exit_code,
        }
    )
    write_json(RESULTS_DIR / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["official_reproduction", "regional_robustness"], required=True)
    args = parser.parse_args()

    manifest = load_yaml(CONTRACT_ROOT / "manifest.yaml")
    datasets_cfg = load_yaml(CONTRACT_ROOT / "config" / "datasets.yaml")
    experiment = load_yaml(CONTRACT_ROOT / "config" / "experiment.yaml")
    method_root = Path(experiment.get("method_root", manifest.get("method_entry_root_hint", "/app/PriOr-RAFT")))
    dataset_cfg = (datasets_cfg.get("datasets") or {})[experiment["dataset"]]
    dataset_env = {
        "flowscape": "PRIORFLOW_FLOWSCAPE_ROOT",
        "mpfdataset": "PRIORFLOW_MPFDATASET_ROOT",
        "omniphotos": "PRIORFLOW_OMNIPHOTOS_ROOT",
        "odvista": "PRIORFLOW_ODVISTA_ROOT",
    }.get(experiment["dataset"])
    dataset_root = os.environ.get(dataset_env, dataset_cfg["root"]) if dataset_env else dataset_cfg["root"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(method_root) + os.pathsep + str(method_root / "core") + os.pathsep + env.get("PYTHONPATH", "")
    if dataset_env:
        env[dataset_env] = dataset_root
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    command_cfg = experiment["regional_command"] if args.scenario == "regional_robustness" else experiment["official_command"]
    proc = run_official(method_root, command_cfg["script"], command_cfg.get("args", []), env)
    command = [command_cfg["script"]] + command_cfg.get("args", [])

    if args.scenario == "regional_robustness":
        quality = parse_regional_metrics(proc.stdout)
    else:
        quality = parse_official_metrics(proc.stdout)

    quality["raw_metric_source"] = "PriOr-RAFT/evaluate.py stdout"
    write_json(RESULTS_DIR / "quality_metrics.json", quality)
    update_metadata(manifest, experiment, args.scenario, command, proc.returncode)

    summary = {
        "scenario": args.scenario,
        "command": command,
        "exit_code": proc.returncode,
        "dataset_root": dataset_root,
        "quality_metrics": quality,
    }
    write_json(OUTPUTS_DIR / args.scenario / "inference_summary.json", summary)

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
