"""Fail-closed orchestration for the immutable-input Stage 4 revision."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import h5py
import numpy as np
import pandas as pd
import yaml

from .stage4_audit import (
    EXPECTED_MODEL_IDENTITIES,
    STAGE4_MODEL_DIRECTORIES,
    audit_stage4_inputs,
)
from .design_matrix import lagged_design
from .stage4_compute_scope import (
    PRIMARY_FAST_MODELS,
    load_legacy_fixed_alphas,
    load_stage4_grouped_fixed_alphas,
    validate_legacy_alpha_compatibility,
)
from .stage4_encoding import (
    FAMILIES,
    fit_stage4_encoding,
    fit_stage4_fixed_alpha_grouped,
    load_stage4_recordings,
)
from .stage4_nulls import (
    SUBSTANTIVE_FAMILIES,
    ShortRecordingError,
    circular_shift_null,
    validate_null_count,
)
from .stage4_qc import LAG_CONVENTION
from .stage4_statistics import (
    analyze_stage4_statistics,
    benjamini_hochberg,
    exact_paired_sign_flip,
    recording_cluster_bootstrap_ci,
)


VARIANTS = ("original", "rich", "capacity", "zero_lag")
REQUIRED_FIT_ARTIFACTS = (
    "scores.csv",
    "split_reports.json",
    "pca_reports.json",
    "capacity_reports.json",
)
PREDICTION_ARTIFACT = "predictions.npz"
STATUS_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def load_stage4_config(path: str | Path = "configs/stage4_revision.yaml") -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage 4 config must be a mapping")
    configured = [entry.get("directory") for entry in config.get("models", [])]
    if configured != list(STAGE4_MODEL_DIRECTORIES):
        raise ValueError(
            "Stage 4 config must contain the exact ordered nine-model manifest"
        )
    if len(configured) != len(set(configured)) or any(
        str(value).lower().startswith("bert") for value in configured
    ):
        raise ValueError("Stage 4 model manifest is duplicated or contains BERT")
    identities = {
        entry.get("directory"): entry.get("identity") for entry in config["models"]
    }
    if identities != EXPECTED_MODEL_IDENTITIES:
        raise ValueError("Stage 4 model identities differ from the fixed manifest")
    variants = config.get("variants", {})
    if tuple(variants) != VARIANTS:
        raise ValueError(f"Stage 4 variants must be exactly {list(VARIANTS)}")
    expected = {
        "original": (
            "original", [-0.2, -0.1, 0.0, 0.1, 0.2], False, 5,
            "fits_original_recording_level",
        ),
        "rich": (
            "rich", [-0.2, -0.1, 0.0, 0.1, 0.2], False, None,
            "fits_rich_acoustic",
        ),
        "capacity": (
            "original", [-0.2, -0.1, 0.0, 0.1, 0.2], True, None,
            "fits_capacity_matched",
        ),
        "zero_lag": ("original", [0.0], False, None, "zero_lag"),
    }
    for name, (
        feature_set,
        lags,
        capacity,
        sensitivity_folds,
        output_subdir,
    ) in expected.items():
        observed = variants[name]
        if (
            observed.get("feature_set") != feature_set
            or [float(value) for value in observed.get("lags_seconds", [])] != lags
            or bool(observed.get("capacity_mode")) != capacity
            or observed.get("sensitivity_folds") != sensitivity_folds
            or observed.get("output_subdir") != output_subdir
        ):
            raise ValueError(f"Invalid fixed definition for variant {name!r}")
    if list(config.get("families", [])) != list(FAMILIES):
        raise ValueError(f"families must be exactly {list(FAMILIES)}")
    if int(config.get("expected_recording_count", 0)) < 3:
        raise ValueError("expected_recording_count must be at least three")
    if set(config.get("governance_inputs", {})) != {
        "revision_roadmap",
        "claim_surface_manifest",
    }:
        raise ValueError("Stage 4 governance input paths are required")
    if int(config.get("nulls", {}).get("production_count", 0)) != 100:
        raise ValueError("Stage 4 production null count must be exactly 100")
    if int(config["nulls"].get("dry_run_count", 0)) < 20:
        raise ValueError("Stage 4 dry-run null count must be at least 20")
    return config


def _safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "\x00" in value:
        raise ValueError(f"Unsafe {label}: {value!r}")
    return quote(value, safe="._-")


def discover_layers(activation_store: str | Path) -> list[str]:
    """Discover the sorted intersection of complete canonical HDF5 layers."""
    with h5py.File(activation_store, "r") as store:
        recording_ids = sorted(name for name in store if not name.startswith("__"))
        if not recording_ids:
            raise ValueError(f"No recordings in activation store {activation_store}")
        layer_sets = []
        for recording_id in recording_ids:
            group = store[recording_id]
            if not bool(group.attrs.get("complete", False)):
                raise ValueError(f"{recording_id}: activation group is incomplete")
            if "canonical" not in group:
                raise ValueError(f"{recording_id}: canonical activation group is missing")
            try:
                names = json.loads(group.attrs["layer_names_json"])
            except (KeyError, TypeError, json.JSONDecodeError):
                names = []
                group["canonical"].visititems(
                    lambda name, value: names.append(name)
                    if isinstance(value, h5py.Dataset)
                    else None
                )
            if (
                not isinstance(names, list)
                or not names
                or not all(
                    isinstance(name, str)
                    and isinstance(group["canonical"].get(name), h5py.Dataset)
                    for name in names
                )
            ):
                raise ValueError(f"{recording_id}: canonical layer manifest is invalid")
            layer_sets.append(set(names))
        common = set.intersection(*layer_sets)
        union = set.union(*layer_sets)
        if not common or common != union:
            raise ValueError("Canonical layer schema differs between recordings")
        return sorted(common)


def _git_details(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _package_versions() -> dict[str, str | None]:
    names = ("speech-strf", "numpy", "pandas", "scikit-learn", "scipy", "h5py", "pyyaml")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _artifact_schema(path: Path) -> tuple[bool, str]:
    try:
        if path.name == "scores.csv":
            frame = pd.read_csv(path)
            required = {
                "model",
                "layer",
                "variant",
                "split_kind",
                "recording_id",
                "family",
                "delta_r2",
            }
            if required - set(frame) or frame.empty:
                return False, "scores_schema_invalid"
            numeric = pd.to_numeric(frame["delta_r2"], errors="coerce").to_numpy()
            if not np.isfinite(numeric).all():
                return False, "scores_nonfinite"
        elif path.name == "predictions.npz":
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    return False, "predictions_empty"
                for key in archive.files:
                    value = archive[key]
                    if value.ndim != 2 or not np.isfinite(value).all():
                        return False, f"prediction_invalid:{key}"
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, (list, dict)):
                return False, "json_schema_invalid"
    except Exception as exc:
        return False, f"unreadable:{type(exc).__name__}:{exc}"
    return True, "ok"


def validate_unit(
    unit: str | Path,
    *,
    expected_source_hashes: Mapping[str, str] | None = None,
    expected_config_hash: str | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validate status, provenance, artifact hashes, and artifact schemas."""
    root = Path(unit)
    status_path = root / "status.json"
    if not status_path.is_file():
        return False, "status_missing"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"status_unreadable:{exc}"
    if (
        not isinstance(status, dict)
        or status.get("schema_version") != STATUS_SCHEMA_VERSION
        or status.get("state") != "complete"
    ):
        return False, "status_incomplete_or_schema_invalid"
    if expected_config_hash is not None and status.get("config_sha256") != expected_config_hash:
        return False, "config_hash_mismatch"
    if expected_metadata is not None:
        for key, value in expected_metadata.items():
            if status.get(key) != value:
                return False, f"unit_identity_mismatch:{key}"
    if expected_source_hashes is not None and status.get("source_hashes") != dict(
        sorted(expected_source_hashes.items())
    ):
        return False, "source_hash_mismatch"
    artifacts = status.get("artifacts")
    expected_artifacts = set(REQUIRED_FIT_ARTIFACTS)
    if status.get("null_index") is None:
        expected_artifacts.add(PREDICTION_ARTIFACT)
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        return False, "artifact_manifest_invalid"
    for name in sorted(expected_artifacts):
        path = root / name
        if not path.is_file():
            return False, f"artifact_missing:{name}"
        if sha256_file(path) != artifacts[name].get("sha256"):
            return False, f"artifact_hash_mismatch:{name}"
        valid, reason = _artifact_schema(path)
        if not valid:
            return False, f"artifact_schema_invalid:{name}:{reason}"
    return True, "ok"


def _preserve_corrupt(destination: Path, reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    preserved = destination.with_name(f"{destination.name}.corrupt-{stamp}")
    os.replace(destination, preserved)
    _write_json(
        preserved / "corruption_diagnostic.json",
        {
            "diagnosed_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "original_path": str(destination),
            "preserved_path": str(preserved),
        },
    )
    return preserved


def _prediction_arrays(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split_name in ("predictions", "sensitivity_predictions"):
        split = result[split_name]
        for recording_id in sorted(split):
            for family in sorted(split[recording_id]):
                for kind in ("target", "full", "reduced"):
                    key = "::".join((split_name, recording_id, family, kind))
                    arrays[key] = np.asarray(split[recording_id][family][kind])
    return arrays


def _publish_fit_unit(
    destination: Path,
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    config_hash: str,
    repository_root: Path,
    *,
    include_predictions: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        scores = result["scores"].copy()
        for key in ("model", "layer", "variant"):
            scores[key] = metadata[key]
        scores.to_csv(temporary / "scores.csv", index=False)
        if include_predictions:
            np.savez_compressed(
                temporary / PREDICTION_ARTIFACT, **_prediction_arrays(result)
            )
        for result_key, filename in (
            ("split_reports", "split_reports.json"),
            ("pca_reports", "pca_reports.json"),
            ("capacity_reports", "capacity_reports.json"),
        ):
            reports = []
            for report in result[result_key]:
                current = dict(report)
                current.update({key: metadata[key] for key in ("model", "layer", "variant")})
                reports.append(current)
            _write_json(temporary / filename, reports)
        artifact_names = list(REQUIRED_FIT_ARTIFACTS)
        if include_predictions:
            artifact_names.append(PREDICTION_ARTIFACT)
        artifacts = {
            name: {
                "sha256": sha256_file(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in artifact_names
        }
        status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "state": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **metadata,
            "config_sha256": config_hash,
            "config_hashes": {
                path.name: sha256_file(path)
                for path in sorted((repository_root / "configs").glob("*.yaml"))
                if path.is_file()
            },
            "source_hashes": dict(sorted(source_hashes.items())),
            "git": _git_details(repository_root),
            "packages": _package_versions(),
            "command": [sys.executable, *sys.argv],
            "integrity_validation": {
                "algorithm": "sha256",
                "schema_validated_before_publish": True,
            },
            "artifacts": artifacts,
        }
        _write_json(temporary / "status.json", status)
        valid, reason = validate_unit(
            temporary,
            expected_source_hashes=source_hashes,
            expected_config_hash=config_hash,
        )
        if not valid:
            raise RuntimeError(f"Temporary unit failed integrity validation: {reason}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class Stage4Runner:
    """Run deterministic Stage 4 fits, nulls, summaries, and synthetic checks."""

    def __init__(self, config_path: str | Path = "configs/stage4_revision.yaml") -> None:
        self.config_path = Path(config_path).resolve()
        self.config = load_stage4_config(self.config_path)
        self.root = self.config_path.parent.parent

        def resolve(value: str | Path) -> Path:
            path = Path(value)
            return path if path.is_absolute() else self.root / path

        self.resolve = resolve
        self.output_root = resolve(self.config["output_root"])
        self.audit_output = resolve(self.config["audit_output"])
        self.config_hash = sha256_file(self.config_path)
        self.model_names = [entry["directory"] for entry in self.config["models"]]
        self._source_hash_cache: dict[tuple[str, int, int], str] = {}
        self._audit_report: dict[str, Any] | None = None

    def _source_hash(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        if key not in self._source_hash_cache:
            self._source_hash_cache[key] = sha256_file(path)
        return self._source_hash_cache[key]

    def _model(self, model: str) -> Path:
        if model not in self.model_names:
            raise ValueError(f"model must be one of {self.model_names}")
        return self.resolve(self.config["model_root"]) / model

    def _audit(self, *, force: bool = False) -> dict[str, Any]:
        audit_path = self.resolve(self.config["audit_output"])
        if audit_path.is_file() and not force:
            try:
                saved = json.loads(audit_path.read_text(encoding="utf-8"))
                same_manifest = (
                    Path(saved["manifest"]["path"]).resolve()
                    == self.resolve(self.config["manifest"]).resolve()
                )
                same_model_root = (
                    Path(saved["model_root"]).resolve()
                    == self.resolve(self.config["model_root"]).resolve()
                )
                if saved.get("complete") and same_manifest and same_model_root:
                    self._audit_report = saved
                    return saved
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        config_paths = {
            "stage4_revision": self.config_path,
            "data": self.root / "configs/data.yaml",
            "models": self.root / "configs/models.yaml",
            "features": self.root / "configs/features.yaml",
            "analysis": self.root / "configs/analysis.yaml",
            **{
                name: self.resolve(path)
                for name, path in self.config["governance_inputs"].items()
            },
        }
        report = audit_stage4_inputs(
            manifest_path=self.resolve(self.config["manifest"]),
            features_dir=self.resolve(self.config["features"]["original"]),
            model_root=self.resolve(self.config["model_root"]),
            output_path=audit_path,
            config_paths=config_paths,
            expected_recording_count=int(self.config["expected_recording_count"]),
        )
        if not report.get("complete"):
            raise RuntimeError(
                f"Stage 4 input audit is incomplete ({len(report.get('errors', []))} errors)"
            )
        self._audit_report = report
        return report

    def audit(self) -> dict[str, Any]:
        """Run the full immutable-input audit and replace only its Stage 4 report."""
        return self._audit(force=True)

    def _recording_ids(self) -> list[str]:
        manifest = pd.read_csv(self.resolve(self.config["manifest"]))
        ids = manifest["recording_id"].astype(str).tolist()
        expected = int(self.config["expected_recording_count"])
        if len(ids) != expected or len(ids) != len(set(ids)):
            raise ValueError(
                f"Stage 4 requires exactly {expected} unique recordings; "
                f"observed {len(ids)} rows and {len(set(ids))} unique IDs"
            )
        return ids

    def _groups(self) -> dict[str, Any] | None:
        manifest = pd.read_csv(self.resolve(self.config["manifest"]))
        for column in ("sensitivity_group", "story_id", "group"):
            if column in manifest:
                return dict(zip(manifest["recording_id"].astype(str), manifest[column]))
        return None

    def _feature_dir(self, variant: str) -> Path:
        feature_set = self.config["variants"][variant]["feature_set"]
        return self.resolve(self.config["features"][feature_set])

    def _fit_root(self, variant: str) -> Path:
        return self.output_root / self.config["variants"][variant]["output_subdir"]

    def _source_hashes(
        self, model: str, feature_dir: Path, recording_ids: Sequence[str]
    ) -> dict[str, str]:
        audit = self._audit_report or self._audit()
        paths = {
            "manifest": self.resolve(self.config["manifest"]),
            **{
                f"model_metadata:{name}": self._model(model) / name
                for name in (
                    "run_metadata.json",
                    "layer_metadata.json",
                    "comparability_contract.json",
                    "extraction_manifest.csv",
                )
            },
            **{
                f"config:{path.name}": path
                for path in sorted((self.root / "configs").glob("*.yaml"))
                if path.is_file()
            },
            **{
                f"governance:{name}": self.resolve(path)
                for name, path in self.config["governance_inputs"].items()
            },
            **{
                f"code:{name}": Path(__file__).with_name(f"{name}.py")
                for name in (
                    "stage4_runner",
                    "stage4_encoding",
                    "stage4_statistics",
                    "stage4_nulls",
                    "stage4_audit",
                    "stage4_compute_scope",
                    "design_matrix",
                    "evaluate",
                )
            },
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing immutable Stage 4 sources: {missing}")
        hashes = {
            name: self._source_hash(path)
            for name, path in sorted(paths.items())
        }
        model_audit = next(
            value for value in audit["models"] if value["directory"] == model
        )
        activation_path = self._model(model) / "activations.h5"
        activation_stat = activation_path.stat()
        if (
            model_audit.get("activations_sha256") is None
            or model_audit.get("activations_bytes") != activation_stat.st_size
            or model_audit.get("activations_mtime_ns") != activation_stat.st_mtime_ns
        ):
            raise RuntimeError(
                f"Immutable activation identity changed after audit: {activation_path}"
            )
        hashes["activations"] = model_audit["activations_sha256"]
        recording_audit = {
            value["recording_id"]: value["feature"] for value in audit["recordings"]
        }
        for recording_id in recording_ids:
            feature_path = feature_dir / f"{recording_id}.npz"
            if feature_dir == self.resolve(self.config["features"]["original"]):
                saved = recording_audit[recording_id]
                stat = feature_path.stat()
                if (
                    saved.get("sha256") is None
                    or saved.get("bytes") != stat.st_size
                    or saved.get("mtime_ns") != stat.st_mtime_ns
                ):
                    raise RuntimeError(
                        f"Immutable feature identity changed after audit: {feature_path}"
                    )
                hashes[f"feature:{recording_id}"] = saved["sha256"]
            else:
                hashes[f"feature:{recording_id}"] = self._source_hash(feature_path)
        return dict(sorted(hashes.items()))

    def _destination(
        self,
        model: str,
        layer: str,
        *,
        variant: str,
        null_index: int | None = None,
        dry_run: bool = False,
    ) -> Path:
        if null_index is None:
            relative = Path(self.config["variants"][variant]["output_subdir"])
        else:
            mode = "dry_nonfinal" if dry_run else "full"
            relative = Path(self.config["output_subdirs"]["nulls"]) / mode
        destination = (
            self.output_root
            / relative
            / _safe_component(model, "model")
            / _safe_component(layer, "layer")
        )
        if null_index is not None:
            destination /= f"null_{null_index:03d}"
        return destination

    def _fit_one(
        self,
        model: str,
        layer: str,
        variant: str,
        *,
        null_index: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        usage_started = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        feature_dir = self._feature_dir(variant)
        ids = self._recording_ids()
        variant_config = self.config["variants"][variant]
        encoding = self.config["encoding"]
        reduced_families = (
            SUBSTANTIVE_FAMILIES if null_index is not None else FAMILIES
        )
        sensitivity_folds = (
            None
            if null_index is not None
            else variant_config.get("sensitivity_folds")
        )
        source_hashes = self._source_hashes(model, feature_dir, ids)
        destination = self._destination(
            model,
            layer,
            variant=variant,
            null_index=null_index,
            dry_run=dry_run,
        )
        valid, reason = validate_unit(
            destination,
            expected_source_hashes=source_hashes,
            expected_config_hash=self.config_hash,
            expected_metadata={
                "model": model,
                "layer": layer,
                "variant": variant,
                "null_index": null_index,
                "null_mode": (
                    None if null_index is None else ("dry_nonfinal" if dry_run else "full")
                ),
                "reduced_families": list(reduced_families),
                "sensitivity_folds": sensitivity_folds,
            },
        )
        if valid:
            return {"state": "resumed", "path": str(destination)}
        preserved = None
        if destination.exists():
            preserved = _preserve_corrupt(destination, reason)

        activation_store = self._model(model) / "activations.h5"
        recordings = load_stage4_recordings(feature_dir, activation_store, layer, recording_ids=ids)
        null_manifest = None
        if null_index is not None:
            lags = self.config["variants"]["original"]["lags_seconds"]
            recordings, null_manifest = circular_shift_null(
                recordings,
                null_index=null_index,
                seed=int(self.config["random_seed"]),
                rate_hz=float(self.config["analysis_rate_hz"]),
                lags_seconds=lags,
                minimum_zero_seconds=float(self.config["nulls"]["minimum_zero_seconds"]),
            )
        result = fit_stage4_encoding(
            recordings,
            alphas=encoding["alphas"],
            lags_seconds=variant_config["lags_seconds"],
            rate_hz=float(self.config["analysis_rate_hz"]),
            target_pca_components=encoding.get("target_pca_components"),
            capacity_mode=bool(variant_config["capacity_mode"]),
            reduced_families=reduced_families,
            sensitivity_groups=self._groups(),
            sensitivity_folds=sensitivity_folds,
            inner_folds=int(encoding["inner_folds"]),
            random_seed=int(self.config["random_seed"]),
        )
        metadata: dict[str, Any] = {
            "model": model,
            "registered_model_identity": next(
                value["resolved_model_identity"]
                for value in (self._audit_report or {})["models"]
                if value["directory"] == model
            ),
            "layer": layer,
            "variant": variant,
            "random_seed": int(self.config["random_seed"]),
            "null_index": null_index,
            "null_mode": (
                None if null_index is None else ("dry_nonfinal" if dry_run else "full")
            ),
            "final_inference_eligible": null_index is None or not dry_run,
            "lags_seconds": list(variant_config["lags_seconds"]),
            "reduced_families": list(reduced_families),
            "sensitivity_folds": sensitivity_folds,
            "lag_convention": LAG_CONVENTION,
            "fit_runtime_seconds": time.perf_counter() - started,
            "peak_rss_kib": max(
                usage_started,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
        }
        if null_manifest is not None:
            metadata["null_shift_manifest"] = null_manifest
        _publish_fit_unit(
            destination,
            result,
            metadata,
            source_hashes,
            self.config_hash,
            self.root,
            include_predictions=null_index is None,
        )
        return {
            "state": "computed",
            "path": str(destination),
            "preserved_corrupt": str(preserved) if preserved else None,
        }

    def fit(
        self, model: str, variant: str, layer: str | None = None
    ) -> list[dict[str, Any]]:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")
        self._model(model)
        self._audit()
        available = discover_layers(self._model(model) / "activations.h5")
        selected = available if layer is None else [layer]
        if any(value not in available for value in selected):
            raise ValueError(f"Requested layer is absent; available layers: {available}")
        return [self._fit_one(model, value, variant) for value in selected]

    def null(
        self,
        model: str,
        null_index: int,
        *,
        dry_run: bool = False,
        null_count: int | None = None,
        layer: str | None = None,
    ) -> list[dict[str, Any]]:
        count = int(
            null_count
            if null_count is not None
            else self.config["nulls"]["dry_run_count" if dry_run else "production_count"]
        )
        validate_null_count(count, dry_run=dry_run)
        if null_index < 0 or null_index >= count:
            raise ValueError(f"null_index must be in [0, {count})")
        self._model(model)
        self._audit()
        available = discover_layers(self._model(model) / "activations.h5")
        selected = available if layer is None else [layer]
        if any(value not in available for value in selected):
            raise ValueError(f"Requested layer is absent; available layers: {available}")
        return [
            self._fit_one(
                model, value, "original", null_index=null_index, dry_run=dry_run
            )
            for value in selected
        ]

    def _fast_refit_destination(
        self, model: str, layer: str, namespace: str = "primary_fast_refits"
    ) -> Path:
        return (
            self.output_root
            / _safe_component(namespace, "output namespace")
            / _safe_component(model, "model")
            / _safe_component(layer, "layer")
        )

    def _fixed_null_destination(
        self, model: str, layer: str, null_index: int
    ) -> Path:
        return (
            self.output_root
            / "null_controls"
            / "fixed_alpha_20"
            / _safe_component(model, "model")
            / _safe_component(layer, "layer")
            / f"null_{null_index:03d}"
        )

    def _legacy_alpha_source(
        self, model: str, layer: str
    ) -> tuple[dict[int, dict[str, float]], Path, Path]:
        contract, results = validate_legacy_alpha_compatibility(
            self._model(model),
            manifest_path=self.resolve(self.config["manifest"]),
            features_dir=self.resolve(self.config["features"]["original"]),
            feature_config_path=self.root / "configs" / "features.yaml",
            analysis_config_path=self.root / "configs" / "analysis.yaml",
        )
        return load_legacy_fixed_alphas(results, layer), contract, results

    def fast_refit(
        self,
        model: str,
        layer: str,
        *,
        output_namespace: str = "primary_fast_refits",
    ) -> dict[str, Any]:
        """Run one all-recording grouped refit with frozen legacy alphas."""
        if model not in PRIMARY_FAST_MODELS:
            raise ValueError(
                "Fast primary refits are restricted to the eight non-HuBERT-Base "
                f"pilot checkpoints: {list(PRIMARY_FAST_MODELS)}"
            )
        started = time.perf_counter()
        self._audit()
        available = discover_layers(self._model(model) / "activations.h5")
        if layer not in available:
            raise ValueError(f"Requested layer is absent; available layers: {available}")
        fixed_alphas, contract_path, results_path = self._legacy_alpha_source(
            model, layer
        )
        ids = self._recording_ids()
        feature_dir = self._feature_dir("original")
        source_hashes = self._source_hashes(model, feature_dir, ids)
        source_hashes.update(
            {
                "fixed_alpha_contract": self._source_hash(contract_path),
                "fixed_alpha_results": self._source_hash(results_path),
            }
        )
        destination = self._fast_refit_destination(
            model, layer, namespace=output_namespace
        )
        identity = {
            "model": model,
            "layer": layer,
            "variant": "primary_fast_refit",
            "null_index": None,
            "null_mode": None,
            "reduced_families": list(FAMILIES),
            "sensitivity_folds": 5,
            "inner_cv_repeated": False,
            "output_namespace": output_namespace,
            "target_pca_solver": "auto_matches_legacy_original",
        }
        valid, reason = validate_unit(
            destination,
            expected_source_hashes=source_hashes,
            expected_config_hash=self.config_hash,
            expected_metadata=identity,
        )
        if valid:
            return {"state": "resumed", "path": str(destination)}
        preserved = (
            _preserve_corrupt(destination, reason) if destination.exists() else None
        )
        recordings = load_stage4_recordings(
            feature_dir,
            self._model(model) / "activations.h5",
            layer,
            recording_ids=ids,
        )
        result = fit_stage4_fixed_alpha_grouped(
            recordings,
            fixed_alphas=fixed_alphas,
            lags_seconds=self.config["variants"]["original"]["lags_seconds"],
            rate_hz=float(self.config["analysis_rate_hz"]),
            target_pca_components=self.config["encoding"].get(
                "target_pca_components"
            ),
            reduced_families=FAMILIES,
            sensitivity_groups=None,
            outer_folds=5,
            random_seed=int(self.config["random_seed"]),
            alpha_source=str(results_path),
            target_pca_solver="auto",
        )
        metadata = {
            **identity,
            "registered_model_identity": next(
                value["resolved_model_identity"]
                for value in (self._audit_report or {})["models"]
                if value["directory"] == model
            ),
            "random_seed": int(self.config["random_seed"]),
            "lags_seconds": list(
                self.config["variants"]["original"]["lags_seconds"]
            ),
            "lag_convention": LAG_CONVENTION,
            "alpha_source": str(results_path),
            "alpha_source_contract": str(contract_path),
            "alpha_reuse_compatibility": "passed_exact_comparability_contract",
            "target_pca_solver": "auto_matches_legacy_original",
            "hyperparameter_policy": (
                "fixed legacy original alpha per model/layer/grouped fold/model; "
                "inner CV not repeated"
            ),
            "execution_role": (
                "primary"
                if output_namespace == "primary_fast_refits"
                else "worker_benchmark"
            ),
            "fit_runtime_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        _publish_fit_unit(
            destination,
            result,
            metadata,
            source_hashes,
            self.config_hash,
            self.root,
            include_predictions=True,
        )
        return {
            "state": "computed",
            "path": str(destination),
            "preserved_corrupt": str(preserved) if preserved else None,
        }

    def fixed_alpha_null(
        self, model: str, null_index: int, layer: str
    ) -> dict[str, Any]:
        """Run one model × shift × layer fixed-hyperparameter null unit."""
        from .stage4_compute_scope import NULL_MODELS

        if model not in NULL_MODELS:
            raise ValueError(f"Fixed-alpha null model must be one of {list(NULL_MODELS)}")
        if null_index < 0 or null_index >= 20:
            raise ValueError("Fixed-alpha null_index must be in [0, 20)")
        started = time.perf_counter()
        self._audit()
        available = discover_layers(self._model(model) / "activations.h5")
        if layer not in available:
            raise ValueError(f"Requested layer is absent; available layers: {available}")
        if model == "hubert_base":
            alpha_unit = self._destination(model, layer, variant="original")
            valid, reason = validate_unit(alpha_unit)
            split_kinds = ("sensitivity",)
        else:
            alpha_unit = self._fast_refit_destination(model, layer)
            valid, reason = validate_unit(alpha_unit)
            split_kinds = ("primary_fast_grouped",)
        if not valid:
            raise RuntimeError(
                f"Observed fixed-alpha source is incomplete for {model}/{layer}: {reason}"
            )
        alpha_report = alpha_unit / "split_reports.json"
        fixed_alphas = load_stage4_grouped_fixed_alphas(
            alpha_report,
            reduced_families=SUBSTANTIVE_FAMILIES,
            split_kinds=split_kinds,
        )
        ids = self._recording_ids()
        feature_dir = self._feature_dir("original")
        source_hashes = self._source_hashes(model, feature_dir, ids)
        source_hashes["fixed_alpha_split_reports"] = self._source_hash(alpha_report)
        destination = self._fixed_null_destination(model, layer, null_index)
        identity = {
            "model": model,
            "layer": layer,
            "variant": "original_fixed_alpha_null",
            "null_index": null_index,
            "null_mode": "fixed_alpha_20",
            "reduced_families": list(SUBSTANTIVE_FAMILIES),
            "sensitivity_folds": 5,
            "inner_cv_repeated": False,
            "target_pca_solver": (
                "full_matches_stage4_observed"
                if model == "hubert_base"
                else "auto_matches_fast_refit_and_legacy_original"
            ),
        }
        valid, reason = validate_unit(
            destination,
            expected_source_hashes=source_hashes,
            expected_config_hash=self.config_hash,
            expected_metadata=identity,
        )
        if valid:
            return {"state": "resumed", "path": str(destination)}
        preserved = (
            _preserve_corrupt(destination, reason) if destination.exists() else None
        )
        recordings = load_stage4_recordings(
            feature_dir,
            self._model(model) / "activations.h5",
            layer,
            recording_ids=ids,
        )
        shifted, null_manifest = circular_shift_null(
            recordings,
            null_index=null_index,
            seed=int(self.config["random_seed"]),
            rate_hz=float(self.config["analysis_rate_hz"]),
            lags_seconds=self.config["variants"]["original"]["lags_seconds"],
            minimum_zero_seconds=float(self.config["nulls"]["minimum_zero_seconds"]),
        )
        result = fit_stage4_fixed_alpha_grouped(
            shifted,
            fixed_alphas=fixed_alphas,
            lags_seconds=self.config["variants"]["original"]["lags_seconds"],
            rate_hz=float(self.config["analysis_rate_hz"]),
            target_pca_components=self.config["encoding"].get(
                "target_pca_components"
            ),
            reduced_families=SUBSTANTIVE_FAMILIES,
            sensitivity_groups=None,
            outer_folds=5,
            random_seed=int(self.config["random_seed"]),
            alpha_source=str(alpha_report),
            target_pca_solver=("full" if model == "hubert_base" else "auto"),
        )
        source_split_reports = {
            int(value["outer_fold"]): {
                "train_recording_ids": sorted(value["train_recording_ids"]),
                "test_recording_ids": sorted(value["test_recording_ids"]),
            }
            for value in json.loads(alpha_report.read_text(encoding="utf-8"))
            if value.get("split_kind") in split_kinds
        }
        refit_split_reports = {
            int(value["outer_fold"]): {
                "train_recording_ids": sorted(value["train_recording_ids"]),
                "test_recording_ids": sorted(value["test_recording_ids"]),
            }
            for value in result["split_reports"]
        }
        if source_split_reports != refit_split_reports:
            raise RuntimeError(
                "Fixed-alpha null split differs from its observed alpha source"
            )
        metadata = {
            **identity,
            "registered_model_identity": next(
                value["resolved_model_identity"]
                for value in (self._audit_report or {})["models"]
                if value["directory"] == model
            ),
            "random_seed": int(self.config["random_seed"]),
            "lags_seconds": list(
                self.config["variants"]["original"]["lags_seconds"]
            ),
            "lag_convention": LAG_CONVENTION,
            "null_shift_manifest": null_manifest,
            "alpha_source": str(alpha_report),
            "target_pca_solver": (
                "full_matches_stage4_observed"
                if model == "hubert_base"
                else "auto_matches_fast_refit_and_legacy_original"
            ),
            "hyperparameter_policy": (
                "fixed observed-analysis alpha; inner CV not repeated; "
                "fixed-hyperparameter null sensitivity"
            ),
            "final_inference_eligible": False,
            "fit_runtime_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        _publish_fit_unit(
            destination,
            result,
            metadata,
            source_hashes,
            self.config_hash,
            self.root,
            include_predictions=False,
        )
        return {
            "state": "computed",
            "path": str(destination),
            "preserved_corrupt": str(preserved) if preserved else None,
        }

    def _completed_fit_frames(
        self, variant: str, *, require_complete: bool = True
    ) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        root = self._fit_root(variant)
        recording_ids = self._recording_ids()
        feature_dir = self._feature_dir(variant)
        for model in self.model_names:
            source_hashes = self._source_hashes(model, feature_dir, recording_ids)
            model_root = root / _safe_component(model, "model")
            layers = discover_layers(self._model(model) / "activations.h5")
            for layer in layers:
                unit = model_root / _safe_component(layer, "layer")
                valid, reason = validate_unit(
                    unit,
                    expected_source_hashes=source_hashes,
                    expected_config_hash=self.config_hash,
                    expected_metadata={
                        "model": model,
                        "layer": layer,
                        "variant": variant,
                        "null_index": None,
                        "null_mode": None,
                    },
                )
                if valid:
                    frames.append(pd.read_csv(unit / "scores.csv"))
                else:
                    failures.append(f"{model}/{layer}: {reason}")
        if require_complete and failures:
            raise RuntimeError(
                f"Stage 4 {variant} fit set is incomplete: " + "; ".join(failures)
            )
        return frames

    def summarize(self) -> dict[str, str]:
        """Summarize only integrity-valid completed units into source tables."""
        self._audit()
        summary_dir = self.output_root / self.config["output_subdirs"]["summaries"]
        figure_dir = self.output_root / self.config["output_subdirs"]["figure_sources"]
        summary_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        expected = self._recording_ids()
        bootstrap = int(self.config["statistics"]["bootstrap_samples"])
        seed = int(self.config["random_seed"])
        outputs: dict[str, str] = {}
        variant_primary: dict[str, pd.DataFrame] = {}
        full_r2_recordings: list[pd.DataFrame] = []

        for variant in VARIANTS:
            frames = self._completed_fit_frames(variant)
            if not frames:
                continue
            scores = pd.concat(frames, ignore_index=True)
            primary = scores.query("split_kind == 'primary'").copy()
            primary["effect"] = primary["delta_r2"]
            table = analyze_stage4_statistics(
                primary,
                expected,
                FAMILIES,
                n_bootstrap=bootstrap,
                seed=seed,
            )
            table.insert(0, "variant", variant)
            path = summary_dir / f"primary_{variant}.csv"
            table.to_csv(path, index=False)
            table.to_csv(figure_dir / f"primary_{variant}.csv", index=False)
            outputs[f"primary_{variant}"] = str(path)
            variant_primary[variant] = primary
            full_r2_recordings.append(
                primary[
                    [
                        "variant",
                        "model",
                        "layer",
                        "recording_id",
                        "frames",
                        "duration_seconds",
                        "full_r2",
                    ]
                ].drop_duplicates(
                    ["variant", "model", "layer", "recording_id"]
                )
            )

        if full_r2_recordings:
            full_recording = pd.concat(full_r2_recordings, ignore_index=True)
            full_recording["interpretation"] = (
                "non-ranking diagnostic; compare only with commensurate target-PCA coverage"
            )
            recording_path = summary_dir / "full_model_r2_by_recording.csv"
            full_recording.to_csv(recording_path, index=False)
            full_summary = full_recording.groupby(
                ["variant", "model", "layer"], as_index=False
            ).agg(
                equal_recording_mean_full_r2=("full_r2", "mean"),
                recording_sd=("full_r2", "std"),
                recording_min=("full_r2", "min"),
                recording_max=("full_r2", "max"),
                recording_count=("recording_id", "nunique"),
            )
            full_summary["interpretation"] = (
                "non-ranking diagnostic; target-PCA coverage must be checked"
            )
            summary_path = summary_dir / "full_model_r2_diagnostic.csv"
            full_summary.to_csv(summary_path, index=False)
            outputs["full_r2_recording"] = str(recording_path)
            outputs["full_r2_diagnostic"] = str(summary_path)

        original_frames = self._completed_fit_frames("original")
        if original_frames:
            original = pd.concat(original_frames, ignore_index=True)
            sensitivity = (
                original.groupby(
                    ["model", "recording_id", "family", "split_kind"],
                    as_index=False,
                )
                .agg(
                    delta_r2=("delta_r2", "mean"),
                    duration_seconds=("duration_seconds", "first"),
                )
            )
            comparison = (
                sensitivity.pivot_table(
                    index=["model", "recording_id", "family"],
                    columns="split_kind",
                    values="delta_r2",
                )
                .reset_index()
                .rename(columns={"primary": "loro_delta_r2", "sensitivity": "grouped_delta_r2"})
            )
            comparison["grouped_minus_loro"] = (
                comparison.get("grouped_delta_r2") - comparison.get("loro_delta_r2")
            )
            path = summary_dir / "sensitivity_original_grouped_vs_loro.csv"
            comparison.to_csv(path, index=False)
            comparison.to_csv(figure_dir / path.name, index=False)
            outputs["sensitivity"] = str(path)
            inference_tables = {}
            for split_kind in ("primary", "sensitivity"):
                split = original.query("split_kind == @split_kind").copy()
                split["effect"] = split["delta_r2"]
                inference = analyze_stage4_statistics(
                    split,
                    expected,
                    FAMILIES,
                    n_bootstrap=bootstrap,
                    seed=seed,
                )
                inference_tables[split_kind] = inference[
                    ["model", "family", "effect", "ci_low", "ci_high", "q_value"]
                ].rename(
                    columns={
                        column: f"{split_kind}_{column}"
                        for column in ("effect", "ci_low", "ci_high", "q_value")
                    }
                )
            conclusions = inference_tables["primary"].merge(
                inference_tables["sensitivity"],
                on=["model", "family"],
                validate="one_to_one",
            )
            for split_kind in ("primary", "sensitivity"):
                conclusions[f"{split_kind}_direction"] = np.sign(
                    conclusions[f"{split_kind}_effect"]
                ).astype(int)
                conclusions[f"{split_kind}_fdr_significant"] = (
                    conclusions[f"{split_kind}_q_value"] < 0.05
                )
            conclusions["conclusion_changed"] = (
                conclusions["primary_direction"]
                != conclusions["sensitivity_direction"]
            ) | (
                conclusions["primary_fdr_significant"]
                != conclusions["sensitivity_fdr_significant"]
            )
            conclusion_path = summary_dir / "sensitivity_conclusions.csv"
            conclusions.to_csv(conclusion_path, index=False)
            outputs["sensitivity_conclusions"] = str(conclusion_path)

        pca_rows: list[dict[str, Any]] = []
        capacity_rows: list[dict[str, Any]] = []
        for variant in VARIANTS:
            root = self._fit_root(variant)
            if not root.is_dir():
                continue
            for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                for unit in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                    valid, _ = validate_unit(unit)
                    if not valid:
                        continue
                    for filename, rows in (
                        ("pca_reports.json", pca_rows),
                        ("capacity_reports.json", capacity_rows),
                    ):
                        for report in json.loads((unit / filename).read_text()):
                            if report.get("split_kind") == "primary":
                                rows.append(report)
        if pca_rows:
            pca = pd.DataFrame(pca_rows)
            pca_dir = self.output_root / self.config["output_subdirs"]["pca_coverage"]
            pca_dir.mkdir(parents=True, exist_ok=True)
            pca_fold = pca.copy()
            pca_fold["cumulative_explained_variance_ratio"] = pca_fold[
                "cumulative_explained_variance_ratio"
            ].map(json.dumps)
            fold_path = summary_dir / "pca_coverage_by_outer_fold.csv"
            pca_fold.to_csv(fold_path, index=False)
            pca_fold.to_csv(pca_dir / fold_path.name, index=False)
            outputs["pca_coverage_by_fold"] = str(fold_path)
            coverage = (
                pca.groupby(["variant", "model", "layer"], as_index=False)
                .agg(
                    target_units_before_pca=("target_units_before_pca", "first"),
                    requested_components=("requested_components", "first"),
                    achieved_components_min=("achieved_components", "min"),
                    achieved_components_max=("achieved_components", "max"),
                    coverage_mean=("total_explained_variance_ratio", "mean"),
                    coverage_sd=("total_explained_variance_ratio", "std"),
                    coverage_min=("total_explained_variance_ratio", "min"),
                    coverage_max=("total_explained_variance_ratio", "max"),
                    outer_folds=("outer_fold", "nunique"),
                )
            )
            path = summary_dir / "pca_coverage.csv"
            coverage.to_csv(path, index=False)
            coverage.to_csv(figure_dir / path.name, index=False)
            coverage.to_csv(pca_dir / path.name, index=False)
            outputs["pca_coverage"] = str(path)
        if capacity_rows:
            path = summary_dir / "capacity_reports.json"
            _write_json(path, capacity_rows)
            flat_capacity = []
            for report in capacity_rows:
                if not report.get("enabled"):
                    continue
                for family, values in report["families"].items():
                    flat_capacity.append(
                        {
                            "variant": report["variant"],
                            "model": report["model"],
                            "layer": report["layer"],
                            "outer_fold": report["outer_fold"],
                            "family": family,
                            **values,
                            "train_recording_ids": json.dumps(
                                report["train_recording_ids"]
                            ),
                            "test_recording_ids": json.dumps(
                                report["test_recording_ids"]
                            ),
                        }
                    )
            flat_path = summary_dir / "capacity_coverage_by_outer_fold.csv"
            pd.DataFrame(flat_capacity).to_csv(flat_path, index=False)
            outputs["capacity"] = str(path)
            outputs["capacity_by_fold"] = str(flat_path)

        if {"original", "zero_lag"} <= set(variant_primary):
            keys = ["model", "layer", "recording_id", "family"]
            five = variant_primary["original"].groupby(keys, as_index=False).agg(
                five_lag=("delta_r2", "mean"), duration_seconds=("duration_seconds", "first")
            )
            zero = variant_primary["zero_lag"].groupby(keys, as_index=False).agg(
                zero_lag=("delta_r2", "mean")
            )
            merged = five.merge(zero, on=keys, validate="one_to_one")
            merged["zero_minus_five"] = merged["zero_lag"] - merged["five_lag"]
            recording = merged.groupby(
                ["model", "recording_id", "family"], as_index=False
            ).agg(
                zero_minus_five=("zero_minus_five", "mean"),
                duration_seconds=("duration_seconds", "first"),
            )
            rows = []
            for (model, family), group in recording.groupby(["model", "family"]):
                indexed = group.set_index("recording_id").loc[expected]
                values = indexed["zero_minus_five"].to_numpy()
                test = exact_paired_sign_flip(
                    values,
                    estimand=(
                        "equal-recording mean zero-lag minus five-lag conditional "
                        "delta R2 across benchmark recordings"
                    ),
                )
                low, high = recording_cluster_bootstrap_ci(
                    values, n_bootstrap=bootstrap, seed=seed
                )
                rows.append(
                    {
                        "model": model,
                        "family": family,
                        **test,
                        "ci_low": low,
                        "ci_high": high,
                        "direction": (
                            "zero_lag_higher"
                            if test["effect"] > 0
                            else ("five_lag_higher" if test["effect"] < 0 else "equal")
                        ),
                    }
                )
            table = pd.DataFrame(rows)
            table["q_value"] = np.nan
            for _, indices in table.groupby("model").groups.items():
                table.loc[indices, "q_value"] = benjamini_hochberg(
                    table.loc[indices, "p_value"].to_numpy()
                )
            path = summary_dir / "zero_vs_five_lag.csv"
            table.to_csv(path, index=False)
            table.to_csv(figure_dir / path.name, index=False)
            outputs["zero_vs_five"] = str(path)

        null_table = self._summarize_nulls(expected, bootstrap, seed)
        if not null_table.empty:
            null_path = summary_dir / "null_observed_minus_mean.csv"
            null_table.to_csv(null_path, index=False)
            null_table.to_csv(figure_dir / null_path.name, index=False)
            outputs["null"] = str(null_path)
        manifest = {
            "schema_version": 1,
            "state": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": self.config_hash,
            "random_seed": seed,
            "git": _git_details(self.root),
            "packages": _package_versions(),
            "primary_estimand": (
                "equal-recording mean conditional delta R2 after arithmetic "
                "mean aggregation across all retained layers within each model"
            ),
            "inference_scope": (
                "consistency across the 12 fixed benchmark recordings; not a "
                "population claim about narratives or model families"
            ),
            "checkpoint_independence_assumed": False,
            "outputs": outputs,
            "artifacts": {
                name: {
                    "path": path,
                    "sha256": sha256_file(path),
                }
                for name, path in outputs.items()
            },
        }
        _write_json(summary_dir / "summary_status.json", manifest)
        return outputs

    def _summarize_nulls(
        self, expected: Sequence[str], bootstrap: int, seed: int
    ) -> pd.DataFrame:
        observed_frames = self._completed_fit_frames("original")
        if not observed_frames:
            return pd.DataFrame()
        observed = pd.concat(observed_frames, ignore_index=True).query(
            "split_kind == 'primary'"
        )
        null_root = self.output_root / self.config["output_subdirs"]["nulls"] / "full"
        null_count = int(self.config["nulls"]["production_count"])
        rows: list[pd.DataFrame] = []
        completeness = []
        incomplete: list[str] = []
        recording_ids = self._recording_ids()
        feature_dir = self._feature_dir("original")
        for model in self.model_names:
            model_root = null_root / _safe_component(model, "model")
            source_hashes = self._source_hashes(model, feature_dir, recording_ids)
            for layer in discover_layers(self._model(model) / "activations.h5"):
                layer_root = model_root / _safe_component(layer, "layer")
                indices = []
                layer_frames = []
                invalid_reasons = {}
                for index in range(null_count):
                    unit = layer_root / f"null_{index:03d}"
                    valid, reason = validate_unit(
                        unit,
                        expected_source_hashes=source_hashes,
                        expected_config_hash=self.config_hash,
                        expected_metadata={
                            "model": model,
                            "layer": layer,
                            "variant": "original",
                            "null_index": index,
                            "null_mode": "full",
                        },
                    )
                    if valid:
                        indices.append(index)
                        frame = pd.read_csv(unit / "scores.csv").query(
                            "split_kind == 'primary'"
                        )
                        frame["null_index"] = index
                        layer_frames.append(frame)
                    else:
                        invalid_reasons[index] = reason
                completeness.append(
                    {
                        "model": model,
                        "layer": layer,
                        "valid_null_count": len(indices),
                        "complete_100": indices == list(range(null_count)),
                        "invalid_reasons": json.dumps(invalid_reasons, sort_keys=True),
                    }
                )
                if indices == list(range(null_count)):
                    rows.extend(layer_frames)
                else:
                    incomplete.append(
                        f"{model}/{layer}: {len(indices)}/{null_count} valid"
                    )
        summary_dir = self.output_root / self.config["output_subdirs"]["summaries"]
        pd.DataFrame(completeness).to_csv(
            summary_dir / "null_completeness.csv", index=False
        )
        if incomplete:
            raise RuntimeError(
                "Production circular-shift null set is incomplete: "
                + "; ".join(incomplete)
            )
        nulls = pd.concat(rows, ignore_index=True)
        null_mean = nulls.groupby(
            ["model", "layer", "recording_id", "family"], as_index=False
        ).agg(mean_null=("delta_r2", "mean"), null_count=("null_index", "nunique"))
        values = observed.merge(
            null_mean,
            on=["model", "layer", "recording_id", "family"],
            validate="one_to_one",
        )
        values["effect"] = values["delta_r2"] - values["mean_null"]
        values = values[
            values["family"].isin(("prosodic", "phonetic", "word"))
        ]
        recording_values = values.groupby(
            ["model", "recording_id", "family"], as_index=False
        ).agg(
            effect=("effect", "mean"),
            duration_seconds=("duration_seconds", "first"),
            layer_count=("layer", "nunique"),
        )
        summary_rows = []
        for (model, family), group in recording_values.groupby(
            ["model", "family"], sort=True
        ):
            indexed = group.set_index("recording_id")
            missing = sorted(set(expected) - set(indexed.index))
            if missing:
                raise ValueError(
                    f"Incomplete null contrast for {model}/{family}: {missing}"
                )
            ordered = indexed.loc[list(expected)]
            effect = ordered["effect"].to_numpy(dtype=float)
            test = exact_paired_sign_flip(
                effect,
                estimand=(
                    "equal-recording mean observed minus mean circular-shift-null "
                    "conditional delta R2 across benchmark recordings"
                ),
            )
            low, high = recording_cluster_bootstrap_ci(
                effect, n_bootstrap=bootstrap, seed=seed
            )
            summary_rows.append(
                {
                    "model": model,
                    "family": family,
                    **test,
                    "ci_low": low,
                    "ci_high": high,
                    "null_count": null_count,
                    "layer_aggregation": "arithmetic_mean_within_recording",
                }
            )
        return pd.DataFrame(summary_rows)

    def synthetic_test(self) -> dict[str, Any]:
        """Run a small deterministic in-memory encoding check and persist evidence."""
        started = time.perf_counter()
        rng = np.random.default_rng(int(self.config["random_seed"]))
        families = ["acoustic", "prosodic", "phonetic", "word", "onset"]
        coefficient = rng.normal(size=(5, 3))
        recordings = {}
        for index in range(4):
            matrix = rng.normal(size=(16, 5))
            recordings[f"synthetic_{index}"] = {
                "matrix": matrix,
                "targets": matrix @ coefficient,
                "families": families,
                "times": np.arange(16) / float(self.config["analysis_rate_hz"]),
                "duration_seconds": 16 / float(self.config["analysis_rate_hz"]),
            }
        result = fit_stage4_encoding(
            recordings,
            alphas=[1.0],
            lags_seconds=[0.0],
            sensitivity_folds=None,
            random_seed=int(self.config["random_seed"]),
        )
        passed = (
            len(result["scores"]) == 4 * len(FAMILIES)
            and np.isfinite(result["scores"]["delta_r2"]).all()
            and set(result["predictions"]) == set(recordings)
        )
        report = {
            "passed": bool(passed),
            "recording_count": 4,
            "score_rows": len(result["scores"]),
            "seed": int(self.config["random_seed"]),
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "command": [sys.executable, *sys.argv],
        }
        destination = (
            self.output_root / self.config["output_subdirs"]["synthetic_test"]
        )
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "result.json", report)
        if not passed:
            raise RuntimeError("Synthetic Stage 4 test failed")
        return report

    def functional_smoke(
        self,
        model: str = "hubert_base",
        *,
        layer: str | None = None,
        recording_id: str | None = None,
    ) -> dict[str, Any]:
        """Gate the pilot with minimal synthetic CV and one real recording.

        Nested CV cannot be defined from one recording without violating the
        recording-level leakage boundary. The minimal-alpha synthetic test
        exercises fitting; this real-data portion exercises loading, temporal
        alignment, lagging, a recording-local null shift, and atomic writing.
        """
        started = time.perf_counter()
        self._model(model)
        self._audit()
        synthetic = self.synthetic_test()
        available_layers = discover_layers(self._model(model) / "activations.h5")
        selected_layer = layer or available_layers[0]
        if selected_layer not in available_layers:
            raise ValueError(
                f"Requested layer is absent; available layers: {available_layers}"
            )
        candidate_ids = (
            [recording_id] if recording_id is not None else self._recording_ids()
        )
        feature_dir = self._feature_dir("original")
        activation_store = self._model(model) / "activations.h5"
        selected_id = None
        selected_record = None
        shifted = None
        null_manifest = None
        unsupported: list[dict[str, Any]] = []
        for candidate in candidate_ids:
            if candidate not in self._recording_ids():
                raise ValueError(f"Unknown recording_id {candidate!r}")
            loaded = load_stage4_recordings(
                feature_dir,
                activation_store,
                selected_layer,
                recording_ids=[candidate],
            )
            try:
                shifted, null_manifest = circular_shift_null(
                    loaded,
                    null_index=0,
                    seed=int(self.config["random_seed"]),
                    rate_hz=float(self.config["analysis_rate_hz"]),
                    lags_seconds=self.config["variants"]["original"]["lags_seconds"],
                    minimum_zero_seconds=float(
                        self.config["nulls"]["minimum_zero_seconds"]
                    ),
                )
            except ShortRecordingError as exc:
                unsupported.extend(exc.diagnostics)
                if recording_id is not None:
                    raise
                continue
            selected_id = candidate
            selected_record = loaded[candidate]
            break
        if selected_id is None or selected_record is None or shifted is None:
            raise RuntimeError(
                "No recording supports the configured recording-local null shift: "
                f"{unsupported}"
            )

        rate_hz = float(self.config["analysis_rate_hz"])
        design, lagged_families = lagged_design(
            selected_record["matrix"],
            np.repeat(selected_id, len(selected_record["matrix"])),
            selected_record["families"],
            list(self.config["variants"]["original"]["lags_seconds"]),
            rate_hz,
        )
        if (
            not np.isfinite(design).all()
            or not np.isfinite(selected_record["targets"]).all()
            or set(selected_record["families"]) != set(FAMILIES)
            or design.shape[1]
            != selected_record["matrix"].shape[1]
            * len(self.config["variants"]["original"]["lags_seconds"])
        ):
            raise RuntimeError("Functional smoke design or schema validation failed")
        assert null_manifest is not None
        observed_null_families = set(
            null_manifest["recordings"][selected_id]["families"]
        )
        if observed_null_families != set(SUBSTANTIVE_FAMILIES):
            raise RuntimeError("Functional smoke null-family schema is invalid")

        destination = (
            self.output_root
            / "functional_smoke"
            / _safe_component(model, "model")
            / _safe_component(selected_layer, "layer")
            / _safe_component(selected_id, "recording")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
        )
        try:
            np.savez_compressed(
                temporary / "diagnostics.npz",
                times=selected_record["times"],
                lagged_design_first_rows=design[:5],
                shifted_feature_first_rows=shifted[selected_id]["matrix"][:5],
            )
            report = {
                "schema_version": 1,
                "state": "complete",
                "kind": "functional_smoke",
                "model": model,
                "layer": selected_layer,
                "recording_id": selected_id,
                "recordings_loaded": 1,
                "frames": int(len(selected_record["times"])),
                "target_units": int(selected_record["targets"].shape[1]),
                "feature_columns": int(selected_record["matrix"].shape[1]),
                "lagged_columns": int(design.shape[1]),
                "lagged_family_columns": len(lagged_families),
                "alpha_grid": [1.0],
                "synthetic_nested_cv": synthetic,
                "real_recording_nested_cv": {
                    "state": "not_run",
                    "reason": (
                        "nested recording-level CV is undefined for one recording"
                    ),
                },
                "null_shift_count": 1,
                "null_shift_manifest": null_manifest,
                "unsupported_recordings_skipped": unsupported,
                "runtime_seconds": time.perf_counter() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "command": [sys.executable, *sys.argv],
            }
            _write_json(temporary / "report.json", report)
            report["artifacts"] = {
                name: {
                    "sha256": sha256_file(temporary / name),
                    "bytes": (temporary / name).stat().st_size,
                }
                for name in ("diagnostics.npz", "report.json")
            }
            _write_json(temporary / "status.json", report)
            if destination.exists():
                _preserve_corrupt(destination, "superseded_functional_smoke")
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return report


def run_fit(
    model: str,
    variant: str,
    layer: str | None = None,
    *,
    config_path: str | Path = "configs/stage4_revision.yaml",
) -> list[dict[str, Any]]:
    return Stage4Runner(config_path).fit(model, variant, layer)


def run_null(
    model: str,
    null_index: int,
    *,
    dry_run: bool = False,
    null_count: int | None = None,
    layer: str | None = None,
    config_path: str | Path = "configs/stage4_revision.yaml",
) -> list[dict[str, Any]]:
    return Stage4Runner(config_path).null(
        model,
        null_index,
        dry_run=dry_run,
        null_count=null_count,
        layer=layer,
    )


def run_summarize(
    config_path: str | Path = "configs/stage4_revision.yaml",
) -> dict[str, str]:
    return Stage4Runner(config_path).summarize()


def run_synthetic_test(
    config_path: str | Path = "configs/stage4_revision.yaml",
) -> dict[str, Any]:
    return Stage4Runner(config_path).synthetic_test()
