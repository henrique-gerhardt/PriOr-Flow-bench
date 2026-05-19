from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


CONTRACT_ROOT = Path(os.environ.get("PRIORFLOW_CONTRACT_ROOT", "/app/benchmark_contract"))


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def dataset_is_ready(root: Path, expected_layout: Iterable[str]) -> bool:
    return root.is_dir() and all((root / item).exists() for item in expected_layout)


def run_command(cmd, cwd: Optional[Path] = None) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def download_google_drive(file_id: str, url: str, output: Path, help_text: Optional[str] = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"Using cached download: {output}")
        return
    target = str(output)
    source = url or f"https://drive.google.com/uc?id={file_id}"
    try:
        run_command(["gdown", "--fuzzy", source, "-O", target])
    except subprocess.CalledProcessError as exc:
        if output.exists() and output.stat().st_size == 0:
            output.unlink()
        message = [
            f"Google Drive download failed for {source}.",
            "This usually means the file is no longer public, has exceeded quota, or gdown cannot resolve the confirmation URL.",
        ]
        if help_text:
            message.append(help_text)
        raise RuntimeError("\n".join(message)) from exc


def existing_file(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def archive_candidates(dataset_key: str, configured_archive: Path) -> List[Path]:
    env_name = {
        "flowscape": "PRIORFLOW_FLOWSCAPE_ARCHIVE",
        "mpfdataset": "PRIORFLOW_MPFDATASET_ARCHIVE",
    }.get(dataset_key)
    candidates = []
    if env_name and os.environ.get(env_name):
        candidates.append(Path(os.environ[env_name]))
    if os.environ.get("PRIORFLOW_DATASET_ARCHIVE"):
        candidates.append(Path(os.environ["PRIORFLOW_DATASET_ARCHIVE"]))
    candidates.append(configured_archive)
    candidates.extend(sorted(configured_archive.parent.glob(f"{configured_archive.stem}.*")))
    return candidates


def dataset_download_help(dataset_key: str, root: Path, archive: Path, download: Dict) -> str:
    alternates = download.get("manual_alternates") or []
    lines = [
        f"Dataset '{dataset_key}' is not ready at {root}.",
        f"Recovery: download the dataset archive manually and place it at {archive},",
        "or set PRIORFLOW_DATASET_ARCHIVE=/path/to/archive inside the container.",
        "The container will extract and normalize it on the next run.",
    ]
    dataset_env = {
        "flowscape": "PRIORFLOW_FLOWSCAPE_ARCHIVE",
        "mpfdataset": "PRIORFLOW_MPFDATASET_ARCHIVE",
    }.get(dataset_key)
    if dataset_env:
        lines.insert(3, f"For this dataset you can also set {dataset_env}=/path/to/archive.")
    if alternates:
        lines.append("Official mirror/reference links:")
        lines.extend(f"- {item}" for item in alternates)
    return "\n".join(lines)


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(str(child))
        else:
            child.unlink()


def extract_archive(archive: Path, destination: Path) -> None:
    scratch = destination.parent / f".{destination.name}.extracting"
    if scratch.exists():
        shutil.rmtree(str(scratch))
    scratch.mkdir(parents=True)
    print(f"Extracting {archive} to {scratch}")
    if zipfile.is_zipfile(str(archive)):
        with zipfile.ZipFile(str(archive)) as handle:
            handle.extractall(str(scratch))
    elif tarfile.is_tarfile(str(archive)):
        with tarfile.open(str(archive)) as handle:
            handle.extractall(str(scratch))
    else:
        try:
            run_command(["7z", "x", "-y", f"-o{scratch}", str(archive)])
        except Exception as exc:
            raise RuntimeError(f"Unsupported or unreadable dataset archive: {archive}") from exc

    clear_directory(destination)
    candidates = [
        scratch / "Flow360",
        scratch / "FlowScape" / "Flow360",
    ]
    flow360 = next((candidate for candidate in candidates if candidate.exists()), None)
    if flow360 is not None:
        shutil.move(str(flow360), str(destination / "Flow360"))
    else:
        entries = list(scratch.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            for child in entries[0].iterdir():
                shutil.move(str(child), str(destination / child.name))
        else:
            for child in entries:
                shutil.move(str(child), str(destination / child.name))
    shutil.rmtree(str(scratch))


def ensure_symlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(str(target))) == source:
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(str(target))
        else:
            target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(str(source), str(target))


def ensure_checkpoint(experiment: Dict) -> None:
    target = Path(experiment["checkpoint"])
    env_checkpoint = os.environ.get("PRIORFLOW_CHECKPOINT")
    cache = Path(env_checkpoint or experiment.get("checkpoint_cache") or target)
    download = experiment.get("checkpoint_download") or {}
    if target.exists() and target.stat().st_size > 0:
        print(f"Checkpoint ready: {target}")
        return
    if cache.exists() and cache.stat().st_size > 0:
        ensure_symlink_or_copy(cache, target)
        print(f"Checkpoint linked from cache: {cache} -> {target}")
        return
    if os.environ.get("PRIORFLOW_SKIP_DOWNLOAD") == "1":
        raise FileNotFoundError(f"Checkpoint missing and PRIORFLOW_SKIP_DOWNLOAD=1: {target}")
    if download.get("provider") != "google_drive":
        raise RuntimeError(f"No automatic checkpoint downloader configured for {target}")
    help_text = (
        f"Recovery: download the checkpoint manually and place it at {cache}, "
        "or set PRIORFLOW_CHECKPOINT=/path/to/checkpoint inside the container."
    )
    download_google_drive(download.get("file_id", ""), download.get("url", ""), cache, help_text=help_text)
    ensure_symlink_or_copy(cache, target)
    print(f"Checkpoint ready: {target}")


def ensure_dataset(datasets_cfg: Dict, experiment: Dict) -> None:
    dataset_key = experiment.get("dataset", datasets_cfg.get("default_dataset"))
    dataset_cfg = (datasets_cfg.get("datasets") or {})[dataset_key]
    root_env = {
        "flowscape": "PRIORFLOW_FLOWSCAPE_ROOT",
        "mpfdataset": "PRIORFLOW_MPFDATASET_ROOT",
        "omniphotos": "PRIORFLOW_OMNIPHOTOS_ROOT",
        "odvista": "PRIORFLOW_ODVISTA_ROOT",
    }.get(dataset_key)
    root = Path(os.environ.get(root_env, dataset_cfg["root"])) if root_env else Path(dataset_cfg["root"])
    expected_layout = dataset_cfg.get("expected_layout") or []
    if expected_layout and dataset_is_ready(root, expected_layout):
        print(f"Dataset ready: {root}")
        return
    if os.environ.get("PRIORFLOW_SKIP_DOWNLOAD") == "1":
        raise FileNotFoundError(f"Dataset missing and PRIORFLOW_SKIP_DOWNLOAD=1: {root}")
    download = dataset_cfg.get("download") or {}
    archive = Path(download.get("cache_path", f"/data/downloads/{dataset_key}.archive"))
    manual_archive = existing_file(archive_candidates(dataset_key, archive))
    if manual_archive is not None:
        archive = manual_archive
        print(f"Using existing dataset archive: {archive}")
    else:
        help_text = dataset_download_help(dataset_key, root, archive, download)
        if download.get("provider") != "google_drive":
            raise RuntimeError(help_text)
        download_google_drive(download.get("file_id", ""), download.get("url", ""), archive, help_text=help_text)
    if download.get("extract", True):
        extract_archive(archive, root)
    if expected_layout and not dataset_is_ready(root, expected_layout):
        missing = [item for item in expected_layout if not (root / item).exists()]
        raise FileNotFoundError(f"Dataset extraction completed, but expected paths are missing: {missing}")
    print(f"Dataset ready: {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    runtime = load_yaml(CONTRACT_ROOT / "config" / "runtime.yaml")
    datasets_cfg = load_yaml(CONTRACT_ROOT / "config" / "datasets.yaml")
    experiment = load_yaml(CONTRACT_ROOT / "config" / "experiment.yaml")

    if not runtime.get("download_assets", True):
        print("download_assets=false; validating assets only.")
        os.environ["PRIORFLOW_SKIP_DOWNLOAD"] = "1"

    ensure_checkpoint(experiment)
    if args.scenario in {"official_reproduction", "regional_robustness"}:
        ensure_dataset(datasets_cfg, experiment)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"asset preparation failed: {exc}", file=sys.stderr)
        raise
