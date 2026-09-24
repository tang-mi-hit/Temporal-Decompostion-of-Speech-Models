from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_run_manifest(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    input_manifest_path: str | Path | None = None,
    model_revision: str | None = None,
    extra: dict | None = None,
) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    packages = {}
    for name in ("speech-strf", "numpy", "pandas", "scikit-learn", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    input_hash = None
    if input_manifest_path and Path(input_manifest_path).exists():
        input_hash = hashlib.sha256(Path(input_manifest_path).read_bytes()).hexdigest()
    configured_revision = config.get("model", {}).get("revision")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "random_seed": config.get("random_seed"),
        "git_commit": commit,
        "model_revision": model_revision or configured_revision,
        "input_manifest_sha256": input_hash,
        "package_versions": packages,
        **(extra or {}),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "run_manifest.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def write_model_run_metadata(
    output_dir: str | Path,
    model: dict,
    *,
    manifest_path: str | Path,
    model_config_path: str | Path,
    feature_config_path: str | Path,
    analysis_config_path: str | Path,
    split_definition_hash: str | None,
    random_seed: int,
    extra: dict | None = None,
) -> Path:
    packages = {}
    for name in ("speech-strf", "numpy", "pandas", "scikit-learn", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "git_commit": commit,
        "stimulus_manifest_sha256": sha256_file(manifest_path),
        "model_config_sha256": sha256_file(model_config_path),
        "feature_config_sha256": sha256_file(feature_config_path),
        "analysis_config_sha256": sha256_file(analysis_config_path),
        "split_definition_sha256": split_definition_hash,
        "random_seed": random_seed,
        "package_versions": packages,
        **(extra or {}),
    }
    destination = Path(output_dir) / "run_metadata.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination

