from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


STAGE4_MODEL_DIRECTORIES = (
    "hubert_base",
    "hubert_large_refactor_rerun",
    "wav2vec2_base",
    "wav2vec2_large",
    "wavlm_base_plus",
    "wavlm_large",
    "data2vec_audio_base",
    "xls_r_300m",
    "whisper_medium_encoder",
)
MODEL_DIRECTORIES = STAGE4_MODEL_DIRECTORIES

EXPECTED_MODEL_IDENTITIES = {
    directory: (
        "hubert_large_reference"
        if directory == "hubert_large_refactor_rerun"
        else directory
    )
    for directory in STAGE4_MODEL_DIRECTORIES
}

METADATA_FILENAMES = (
    "run_metadata.json",
    "layer_metadata.json",
    "comparability_contract.json",
)


def _path_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "present": path.is_file()}


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_summary(values: np.ndarray) -> dict[str, Any]:
    finite = bool(np.isfinite(values).all())
    monotonic = bool(values.size < 2 or np.all(np.diff(values) > 0))
    return {
        "frame_count": int(values.size),
        "start_seconds": float(values[0]) if values.size else None,
        "end_seconds": float(values[-1]) if values.size else None,
        "finite": finite,
        "strictly_increasing": monotonic,
    }


def _timestamp_mismatch(
    observed: np.ndarray, expected: np.ndarray, tolerance: float
) -> dict[str, Any]:
    shape_match = observed.shape == expected.shape
    maximum = (
        float(np.max(np.abs(observed - expected)))
        if shape_match and observed.size
        else (0.0 if shape_match else None)
    )
    return {
        "shape_match": bool(shape_match),
        "maximum_absolute_seconds": maximum,
        "tolerance_seconds": tolerance,
        "mismatch": bool(not shape_match or maximum is None or maximum > tolerance),
    }


def _read_json(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level value is not an object")
        return value
    except Exception as exc:
        errors.append(f"corrupt {label} {path}: {exc}")
        return None


def _identity_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    model = metadata.get("model")
    if isinstance(model, dict):
        identity = model.get("key", model.get("model_key"))
        if identity:
            return str(identity)
    observed = metadata.get("observed_model_metadata")
    if isinstance(observed, dict) and observed.get("model_key"):
        return str(observed["model_key"])
    return None


def _dataset_is_finite(dataset: h5py.Dataset, block_rows: int = 4096) -> bool:
    if dataset.ndim == 0:
        return bool(np.isfinite(dataset[()]).all())
    for start in range(0, dataset.shape[0], block_rows):
        if not np.isfinite(dataset[start : start + block_rows]).all():
            return False
    return True


def _load_features(
    recording_id: str, path: Path, timestamp_tolerance: float
) -> tuple[dict[str, Any], np.ndarray | None, list[str]]:
    errors: list[str] = []
    result: dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "complete": False,
        "frame_count": None,
        "duration_seconds": None,
        "timestamps": None,
    }
    if not path.is_file():
        errors.append(f"{recording_id}: missing feature NPZ: {path}")
        return result, None, errors
    try:
        with np.load(path, allow_pickle=False) as feature:
            required = {"matrix", "times", "names", "families"}
            missing = required - set(feature.files)
            if missing:
                raise ValueError(f"missing arrays {sorted(missing)}")
            matrix = np.asarray(feature["matrix"])
            times = np.asarray(feature["times"], dtype=np.float64)
            names = np.asarray(feature["names"])
            families = np.asarray(feature["families"])
            if matrix.ndim != 2:
                raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
            if times.ndim != 1 or len(times) != len(matrix):
                raise ValueError("times must be 1D and match matrix rows")
            if names.ndim != 1 or families.ndim != 1:
                raise ValueError("names and families must be 1D")
            if len(names) != matrix.shape[1] or len(families) != matrix.shape[1]:
                raise ValueError("feature labels must match matrix columns")
            timestamps = _timestamp_summary(times)
            if not timestamps["finite"] or not timestamps["strictly_increasing"]:
                raise ValueError("feature timestamps must be finite and strictly increasing")
            if not np.isfinite(matrix).all():
                raise ValueError("feature matrix contains non-finite values")
            frame_count = int(len(times))
            if frame_count > 1:
                step = float(np.median(np.diff(times)))
                duration = float(times[-1] + step)
            elif frame_count == 1:
                duration = None
            else:
                duration = 0.0
            result.update(
                {
                    "complete": True,
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "frame_count": frame_count,
                    "duration_seconds": duration,
                    "feature_count": int(matrix.shape[1]),
                    "timestamps": timestamps,
                    "timestamp_tolerance_seconds": timestamp_tolerance,
                }
            )
            return result, times, errors
    except Exception as exc:
        errors.append(f"{recording_id}: corrupt feature NPZ {path}: {exc}")
        return result, None, errors


def _extraction_durations(path: Path, errors: list[str]) -> dict[str, float]:
    if not path.is_file():
        errors.append(f"missing extraction manifest: {path}")
        return {}
    try:
        frame = pd.read_csv(path)
        required = {"recording_id", "duration_seconds"}
        if required - set(frame):
            raise ValueError(f"missing columns {sorted(required - set(frame))}")
        if frame["recording_id"].duplicated().any():
            raise ValueError("recording_id values are not unique")
        durations = pd.to_numeric(frame["duration_seconds"], errors="raise")
        if not np.isfinite(durations).all() or (durations < 0).any():
            raise ValueError("duration_seconds values must be finite and nonnegative")
        return dict(zip(frame["recording_id"].astype(str), durations.astype(float)))
    except Exception as exc:
        errors.append(f"corrupt extraction manifest {path}: {exc}")
        return {}


def _audit_group(
    group: h5py.Group,
    recording_id: str,
    feature_times: np.ndarray | None,
    duration: float | None,
    timestamp_tolerance: float,
) -> tuple[dict[str, Any], list[str], str | None]:
    errors: list[str] = []
    result: dict[str, Any] = {
        "recording_id": recording_id,
        "present": True,
        "complete": False,
        "duration_seconds": duration,
        "native_timestamps": None,
        "canonical_timestamps": None,
        "canonical_feature_timestamp_mismatch": None,
        "layers": [],
    }
    if not bool(group.attrs.get("complete", False)):
        errors.append(f"{recording_id}: HDF5 group is not marked complete")
    try:
        metadata = json.loads(group.attrs["metadata_json"])
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json is not an object")
    except Exception as exc:
        errors.append(f"{recording_id}: invalid metadata_json: {exc}")
        metadata = {}
    identity = metadata.get("model_key")
    try:
        layer_names = json.loads(group.attrs["layer_names_json"])
        if (
            not isinstance(layer_names, list)
            or not layer_names
            or len(layer_names) != len(set(layer_names))
            or not all(isinstance(name, str) and name for name in layer_names)
        ):
            raise ValueError("must be a non-empty list of unique strings")
    except Exception as exc:
        errors.append(f"{recording_id}: invalid layer_names_json: {exc}")
        layer_names = []

    timestamp_values: dict[str, np.ndarray | None] = {}
    for name in ("native_timestamps", "canonical_timestamps"):
        try:
            dataset = group[name]
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
                raise ValueError("must be a 1D dataset")
            values = np.asarray(dataset[:], dtype=np.float64)
            summary = _timestamp_summary(values)
            if not summary["finite"] or not summary["strictly_increasing"]:
                raise ValueError("must be finite and strictly increasing")
            result[name] = summary
            timestamp_values[name] = values
        except Exception as exc:
            errors.append(f"{recording_id}: invalid {name}: {exc}")
            timestamp_values[name] = None

    canonical_times = timestamp_values["canonical_timestamps"]
    if canonical_times is not None and feature_times is not None:
        mismatch = _timestamp_mismatch(
            canonical_times, feature_times, timestamp_tolerance
        )
        result["canonical_feature_timestamp_mismatch"] = mismatch
        if mismatch["mismatch"]:
            errors.append(
                f"{recording_id}: canonical timestamps differ from feature timestamps"
            )
    else:
        result["canonical_feature_timestamp_mismatch"] = {
            "shape_match": False,
            "maximum_absolute_seconds": None,
            "tolerance_seconds": timestamp_tolerance,
            "mismatch": True,
        }
        errors.append(f"{recording_id}: canonical/feature timestamps cannot be compared")

    native_times = timestamp_values["native_timestamps"]
    for layer_name in layer_names:
        layer = {
            "name": layer_name,
            "complete": False,
            "native_frames": None,
            "canonical_frames": None,
            "hidden_size": None,
            "native_timestamp_mismatch": True,
            "canonical_timestamp_mismatch": True,
        }
        try:
            native = group["native"][layer_name]
            canonical = group["canonical"][layer_name]
            if native.ndim != 2 or canonical.ndim != 2:
                raise ValueError("native and canonical activation datasets must be 2D")
            if native.shape[1] <= 0 or canonical.shape[1] != native.shape[1]:
                raise ValueError("native/canonical hidden dimensions differ or are empty")
            native_mismatch = native_times is None or native.shape[0] != len(native_times)
            canonical_mismatch = (
                canonical_times is None or canonical.shape[0] != len(canonical_times)
            )
            finite = _dataset_is_finite(native) and _dataset_is_finite(canonical)
            layer.update(
                {
                    "native_frames": int(native.shape[0]),
                    "canonical_frames": int(canonical.shape[0]),
                    "hidden_size": int(native.shape[1]),
                    "native_timestamp_mismatch": bool(native_mismatch),
                    "canonical_timestamp_mismatch": bool(canonical_mismatch),
                    "finite": bool(finite),
                }
            )
            if native_mismatch or canonical_mismatch:
                raise ValueError("activation frame counts differ from timestamps")
            if not finite:
                raise ValueError("activation values contain non-finite data")
            layer["complete"] = True
        except Exception as exc:
            errors.append(f"{recording_id}/{layer_name}: invalid layer: {exc}")
        result["layers"].append(layer)

    metadata_native = metadata.get("native_frame_count")
    metadata_canonical = metadata.get("canonical_frame_count")
    if native_times is not None and metadata_native != len(native_times):
        errors.append(f"{recording_id}: metadata native frame count mismatch")
    if canonical_times is not None and metadata_canonical != len(canonical_times):
        errors.append(f"{recording_id}: metadata canonical frame count mismatch")
    result["metadata_frame_counts"] = {
        "native": metadata_native,
        "canonical": metadata_canonical,
    }
    result["complete"] = not errors
    return result, errors, str(identity) if identity else None


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def audit_stage4_inputs(
    manifest_path: str | Path = "outputs/manifest.csv",
    features_dir: str | Path = "outputs/features",
    model_root: str | Path = "outputs",
    output_path: str | Path = "outputs/stage4_revision/audit/input_audit.json",
    *,
    config_paths: dict[str, str | Path] | None = None,
    timestamp_tolerance_seconds: float = 1e-8,
    expected_recording_count: int | None = None,
) -> dict[str, Any]:
    """Audit all fixed Stage 4 inputs and atomically persist a fail-closed report."""
    manifest_path = Path(manifest_path)
    features_dir = Path(features_dir)
    model_root = Path(model_root)
    output_path = Path(output_path)
    errors: list[str] = []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "manifest": {"path": str(manifest_path), "present": manifest_path.is_file()},
        "features_dir": str(features_dir),
        "model_root": str(model_root),
        "model_directories": list(STAGE4_MODEL_DIRECTORIES),
        "excluded_models": ["bert_base_uncased"],
        "config_paths": {
            name: _path_status(Path(path))
            for name, path in (config_paths or {}).items()
        },
        "recordings": [],
        "models": [],
        "errors": errors,
    }
    try:
        for name, status in payload["config_paths"].items():
            if not status["present"]:
                errors.append(f"missing required configuration/governance input {name}: {status['path']}")
        try:
            manifest = pd.read_csv(manifest_path)
            if "recording_id" not in manifest:
                raise ValueError("missing recording_id column")
            ids = manifest["recording_id"].astype(str).tolist()
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("recording_id values must be non-empty and unique")
            if (
                expected_recording_count is not None
                and len(ids) != expected_recording_count
            ):
                raise ValueError(
                    f"expected {expected_recording_count} recordings, found {len(ids)}"
                )
            payload["manifest"]["recording_count"] = len(ids)
            payload["manifest"]["complete"] = True
        except Exception as exc:
            errors.append(f"missing or corrupt manifest {manifest_path}: {exc}")
            ids = []
            payload["manifest"]["complete"] = False

        feature_times: dict[str, np.ndarray | None] = {}
        recording_results: dict[str, dict[str, Any]] = {}
        for recording_id in ids:
            feature, times, feature_errors = _load_features(
                recording_id,
                features_dir / f"{recording_id}.npz",
                timestamp_tolerance_seconds,
            )
            errors.extend(feature_errors)
            feature_times[recording_id] = times
            recording_results[recording_id] = {
                "recording_id": recording_id,
                "duration_seconds": feature.get("duration_seconds"),
                "feature_path": feature["path"],
                "feature": feature,
                "models": [],
            }

        for directory in STAGE4_MODEL_DIRECTORIES:
            root = model_root / directory
            model_errors: list[str] = []
            artifacts = {
                filename.removesuffix(".json"): _path_status(root / filename)
                for filename in METADATA_FILENAMES
            }
            metadata_values = {
                filename: _read_json(
                    root / filename, model_errors, filename.removesuffix(".json")
                )
                for filename in METADATA_FILENAMES
            }
            duration_path = root / "extraction_manifest.csv"
            durations = _extraction_durations(duration_path, model_errors)
            identities = {
                identity
                for identity in (
                    _identity_from_metadata(metadata_values["run_metadata.json"]),
                    _identity_from_metadata(metadata_values["layer_metadata.json"]),
                )
                if identity
            }
            store_path = root / "activations.h5"
            model_result: dict[str, Any] = {
                "directory": directory,
                "path": str(root),
                "expected_model_identity": EXPECTED_MODEL_IDENTITIES[directory],
                "resolved_model_identity": None,
                "activations_path": str(store_path),
                "activations_sha256": None,
                "activations_bytes": None,
                "activations_mtime_ns": None,
                "artifacts": artifacts,
                "extraction_manifest": _path_status(duration_path),
                "recordings": [],
                "complete": False,
                "errors": model_errors,
            }
            try:
                with h5py.File(store_path, "r") as store:
                    incomplete_groups = sorted(
                        name for name in store if name.startswith("__incomplete__")
                    )
                    if incomplete_groups:
                        model_errors.append(
                            f"incomplete HDF5 groups present: {incomplete_groups}"
                        )
                    extras = sorted(
                        name
                        for name in store
                        if not name.startswith("__") and name not in set(ids)
                    )
                    if extras:
                        model_errors.append(
                            f"activation store has recordings absent from manifest: {extras}"
                        )
                    for recording_id in ids:
                        if recording_id not in store:
                            group_result = {
                                "recording_id": recording_id,
                                "present": False,
                                "complete": False,
                                "duration_seconds": durations.get(recording_id),
                                "native_timestamps": None,
                                "canonical_timestamps": None,
                                "canonical_feature_timestamp_mismatch": {
                                    "mismatch": True
                                },
                                "layers": [],
                            }
                            group_errors = [
                                f"{recording_id}: missing activation HDF5 group"
                            ]
                            group_identity = None
                        else:
                            group_result, group_errors, group_identity = _audit_group(
                                store[recording_id],
                                recording_id,
                                feature_times[recording_id],
                                durations.get(recording_id),
                                timestamp_tolerance_seconds,
                            )
                        group_result["feature_path"] = str(
                            features_dir / f"{recording_id}.npz"
                        )
                        for layer in group_result["layers"]:
                            layer["feature_path"] = group_result["feature_path"]
                        model_errors.extend(group_errors)
                        if group_identity:
                            identities.add(group_identity)
                        model_result["recordings"].append(group_result)
                        recording_results[recording_id]["models"].append(
                            {
                                "directory": directory,
                                **group_result,
                            }
                        )
            except Exception as exc:
                model_errors.append(f"missing or corrupt activation store {store_path}: {exc}")
                for recording_id in ids:
                    unavailable = {
                        "recording_id": recording_id,
                        "present": False,
                        "complete": False,
                        "duration_seconds": durations.get(recording_id),
                        "feature_path": str(features_dir / f"{recording_id}.npz"),
                        "native_timestamps": None,
                        "canonical_timestamps": None,
                        "canonical_feature_timestamp_mismatch": {"mismatch": True},
                        "layers": [],
                    }
                    model_result["recordings"].append(unavailable)
                    recording_results[recording_id]["models"].append(
                        {"directory": directory, **unavailable}
                    )
            if store_path.is_file() and not any(
                "missing or corrupt activation store" in error
                for error in model_errors
            ):
                model_result["activations_sha256"] = _sha256_file(store_path)
                model_result["activations_bytes"] = store_path.stat().st_size
                model_result["activations_mtime_ns"] = store_path.stat().st_mtime_ns
            if len(identities) == 1:
                model_result["resolved_model_identity"] = next(iter(identities))
            elif len(identities) > 1:
                model_errors.append(f"conflicting model identities: {sorted(identities)}")
            else:
                model_errors.append("model identity is absent from metadata")
            if (
                model_result["resolved_model_identity"]
                != model_result["expected_model_identity"]
            ):
                model_errors.append(
                    "model identity mismatch: expected "
                    f"{model_result['expected_model_identity']!r}, observed "
                    f"{model_result['resolved_model_identity']!r}"
                )
            missing_durations = sorted(set(ids) - set(durations))
            if missing_durations:
                model_errors.append(
                    f"extraction manifest lacks recordings: {missing_durations}"
                )
            model_result["complete"] = not model_errors
            errors.extend(f"{directory}: {error}" for error in model_errors)
            payload["models"].append(model_result)

        payload["recordings"] = list(recording_results.values())
        payload["complete"] = not errors
    except Exception as exc:
        errors.append(f"unexpected audit failure: {type(exc).__name__}: {exc}")
        payload["complete"] = False
    finally:
        _atomic_write_json(payload, output_path)
    return payload


run_audit = audit_stage4_inputs
