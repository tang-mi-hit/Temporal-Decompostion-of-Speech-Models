"""Stage 4 provenance, annotation, and temporal-context quality control."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import yaml

from .alignments import Interval, parse_textgrid
from .audio import inspect_audio
from .stage4_audit import EXPECTED_MODEL_IDENTITIES, STAGE4_MODEL_DIRECTORIES


LAG_CONVENTION = (
    "At activation time t, a positive lag tau uses predictor X(t - tau), so "
    "positive tau means the predictor precedes the activation; negative tau "
    "uses future predictor values relative to the activation."
)
REQUIRED_MODEL_ARTIFACTS = (
    "run_metadata.json",
    "layer_metadata.json",
    "comparability_contract.json",
    "extraction_manifest.csv",
    "activations.h5",
)
PHONE_TIERS = frozenset({"phone", "phones", "phoneme", "phonemes"})
WORD_TIERS = frozenset({"word", "words"})
STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
NUMERICAL_ENDPOINT_TOLERANCE_SECONDS = 1e-6


def _status(reasons: Iterable[dict[str, str]]) -> str:
    severities = [reason["severity"] for reason in reasons]
    if "FAIL" in severities:
        return "FAIL"
    if "WARN" in severities:
        return "WARN"
    return "PASS"


def _reason(severity: str, code: str, detail: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "detail": detail}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required Stage 4 source unavailable: {path}") from None
    except Exception as exc:
        raise ValueError(f"Invalid JSON source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid JSON source {path}: top-level value must be an object")
    return value


def _walk_mappings(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _saved_values(sources: Iterable[Any], keys: Iterable[str]) -> list[Any]:
    wanted = tuple(keys)
    values: list[Any] = []
    for source in sources:
        for mapping in _walk_mappings(source):
            for key in wanted:
                if key in mapping and mapping[key] is not None:
                    value = mapping[key]
                    if not any(value == previous for previous in values):
                        values.append(value)
    return values


def _one_saved(
    sources: Iterable[Any], keys: Iterable[str], default: Any = None
) -> tuple[Any, bool]:
    values = _saved_values(sources, keys)
    return (values[0] if len(values) == 1 else default), len(values) == 1


def _preferred_saved(
    sources: Iterable[Any], keys: Iterable[str], default: Any = None
) -> tuple[Any, bool]:
    """Return the first named saved field, preserving conflicts as a value list."""
    source_list = list(sources)
    for key in keys:
        values = _saved_values(source_list, (key,))
        if values:
            return (values[0] if len(values) == 1 else values), True
    return default, False


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _annotation_qc(
    recording_id: str,
    audio_path: Path,
    alignment_path: Path,
    endpoint_tolerance_seconds: float,
) -> dict[str, Any]:
    try:
        audio = inspect_audio(audio_path)
    except Exception as exc:
        raise ValueError(
            f"Audio source unavailable or unreadable for {recording_id}: "
            f"{audio_path}: {exc}"
        ) from exc
    try:
        intervals = parse_textgrid(alignment_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Alignment source unavailable for {recording_id}: {alignment_path}"
        ) from None
    except Exception as exc:
        raise ValueError(
            f"Alignment source unreadable for {recording_id}: "
            f"{alignment_path}: {exc}"
        ) from exc

    duration = float(audio["duration_seconds"])
    phones = [
        row
        for row in intervals
        if row.tier.strip().lower() in PHONE_TIERS and row.label.strip()
    ]
    words = [
        row
        for row in intervals
        if row.tier.strip().lower() in WORD_TIERS and row.label.strip()
    ]
    empty_count = sum(not row.label.strip() for row in intervals)
    reversed_count = sum(row.end < row.start for row in intervals)
    out_of_bounds = [
        row
        for row in intervals
        if row.start < -1e-9
        or row.start > duration + 1e-9
        or row.end < -1e-9
        or row.end > duration + 1e-9
    ]
    tolerated_numerical_overhangs = [
        row
        for row in out_of_bounds
        if row.start >= -1e-9
        and row.start <= duration + 1e-9
        and row.end >= -1e-9
        and row.end <= duration + NUMERICAL_ENDPOINT_TOLERANCE_SECONDS
    ]
    tolerated_empty_overhangs = [
        row
        for row in out_of_bounds
        if not row.label.strip()
        and row.start >= -1e-9
        and row.start <= duration + endpoint_tolerance_seconds
        and row.end >= -1e-9
        and row.end <= duration + endpoint_tolerance_seconds
    ]
    tolerated_out_of_bounds = set(tolerated_numerical_overhangs) | set(
        tolerated_empty_overhangs
    )
    fatal_out_of_bounds_count = sum(
        row not in tolerated_out_of_bounds for row in out_of_bounds
    )
    out_of_bounds_count = len(out_of_bounds)
    overlap_count = 0
    by_tier: dict[str, list[Interval]] = {}
    for row in intervals:
        by_tier.setdefault(row.tier, []).append(row)
    for rows in by_tier.values():
        ordered = sorted(rows, key=lambda row: (row.start, row.end))
        overlap_count += sum(
            current.start < previous.end - 1e-9
            for previous, current in zip(ordered, ordered[1:])
        )
    tier_endpoints = {
        tier: max(row.end for row in rows) for tier, rows in by_tier.items()
    }
    annotation_endpoint = max(tier_endpoints.values(), default=0.0)
    endpoint_mismatch = max(
        (abs(endpoint - duration) for endpoint in tier_endpoints.values()),
        default=duration,
    )
    reasons: list[dict[str, str]] = []
    for count, code in (
        (reversed_count, "reversed_intervals"),
        (fatal_out_of_bounds_count, "out_of_bounds_intervals"),
        (overlap_count, "overlapping_intervals"),
    ):
        if count:
            reasons.append(_reason("FAIL", code, f"{count} interval(s)"))
    if empty_count:
        reasons.append(_reason("WARN", "empty_labels", f"{empty_count} interval(s)"))
    if tolerated_numerical_overhangs:
        reasons.append(
            _reason(
                "WARN",
                "tolerated_numerical_endpoint_overhang",
                f"{len(tolerated_numerical_overhangs)} interval(s)",
            )
        )
    if tolerated_empty_overhangs:
        reasons.append(
            _reason(
                "WARN",
                "tolerated_empty_endpoint_overhang",
                f"{len(tolerated_empty_overhangs)} interval(s)",
            )
        )
    if endpoint_mismatch > endpoint_tolerance_seconds:
        reasons.append(
            _reason(
                "WARN",
                "annotation_audio_endpoint_mismatch",
                f"{endpoint_mismatch:.9g} seconds",
            )
        )
    return {
        "recording_id": recording_id,
        "audio_path": str(audio_path),
        "alignment_path": str(alignment_path),
        "audio_sample_rate_hz": int(audio["sample_rate"]),
        "audio_duration_seconds": duration,
        "phone_feature_family": "phonetic",
        "phonetic_family_definition": (
            "forced-aligned phone onsets and phone-category indicators, "
            "including tone-bearing labels where present"
        ),
        "phone_count": len(phones),
        "distinct_phone_category_count": len({row.label.strip() for row in phones}),
        "phone_category_distribution": dict(
            sorted(pd.Series([row.label.strip() for row in phones]).value_counts().items())
        )
        if phones
        else {},
        "word_count": len(words),
        "empty_label_count": empty_count,
        "out_of_bounds_count": out_of_bounds_count,
        "tolerated_numerical_out_of_bounds_count": len(
            tolerated_numerical_overhangs
        ),
        "tolerated_empty_out_of_bounds_count": len(tolerated_empty_overhangs),
        "fatal_out_of_bounds_count": fatal_out_of_bounds_count,
        "overlapping_count": overlap_count,
        "reversed_count": reversed_count,
        "annotation_endpoint_seconds": annotation_endpoint,
        "tier_endpoint_seconds": dict(sorted(tier_endpoints.items())),
        "annotation_audio_endpoint_mismatch_seconds": endpoint_mismatch,
        "status": _status(reasons),
        "reasons": reasons,
    }


def _model_qc(
    directory: str,
    root: Path,
    recording_ids: list[str],
    registered_hf_model_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [name for name in REQUIRED_MODEL_ARTIFACTS if not (root / name).is_file()]
    if missing:
        paths = [str(root / name) for name in missing]
        raise FileNotFoundError(
            f"Required Stage 4 source(s) unavailable for {directory}: {paths}"
        )
    run_metadata = _load_json(root / "run_metadata.json")
    layer_metadata = _load_json(root / "layer_metadata.json")
    comparability = _load_json(root / "comparability_contract.json")
    try:
        extraction = pd.read_csv(root / "extraction_manifest.csv")
    except Exception as exc:
        raise ValueError(
            f"Invalid extraction manifest {root / 'extraction_manifest.csv'}: {exc}"
        ) from exc
    if "recording_id" not in extraction:
        raise ValueError(
            f"Invalid extraction manifest {root / 'extraction_manifest.csv'}: "
            "missing recording_id"
        )
    extraction_metadata: list[dict[str, Any]] = []
    for column in ("metadata_json", "audio_metadata_json"):
        if column not in extraction:
            continue
        for index, value in extraction[column].items():
            if pd.isna(value) or not str(value).strip():
                continue
            try:
                parsed = json.loads(value)
            except Exception as exc:
                raise ValueError(
                    f"Invalid extraction manifest {root / 'extraction_manifest.csv'}: "
                    f"row {index} {column} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Invalid extraction manifest {root / 'extraction_manifest.csv'}: "
                    f"row {index} {column} is not an object"
                )
            extraction_metadata.append(parsed)

    hdf_metadata: list[dict[str, Any]] = []
    layer_name_sets: list[list[str]] = []
    try:
        with h5py.File(root / "activations.h5", "r") as store:
            absent = sorted(set(recording_ids) - set(store))
            if absent:
                raise ValueError(f"missing recording group(s) {absent}")
            for recording_id in recording_ids:
                group = store[recording_id]
                attribute = (
                    "metadata_json"
                    if "metadata_json" in group.attrs
                    else "model_metadata_json"
                )
                if attribute not in group.attrs:
                    raise ValueError(f"{recording_id}: activation metadata is absent")
                metadata = json.loads(group.attrs[attribute])
                if not isinstance(metadata, dict):
                    raise ValueError(f"{recording_id}: activation metadata is not an object")
                hdf_metadata.append(metadata)
                names = json.loads(group.attrs["layer_names_json"])
                if not isinstance(names, list) or not names:
                    raise ValueError(f"{recording_id}: invalid layer_names_json")
                layer_name_sets.append([str(name) for name in names])
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid activation source {root / 'activations.h5'}: {exc}") from exc

    sources = [
        run_metadata,
        layer_metadata,
        comparability,
        *extraction_metadata,
        *hdf_metadata,
    ]
    identity_candidates = [
        run_metadata.get("model", {}).get("key")
        if isinstance(run_metadata.get("model"), dict)
        else None,
        layer_metadata.get("model", {}).get("key")
        if isinstance(layer_metadata.get("model"), dict)
        else None,
        *_saved_values(extraction_metadata + hdf_metadata, ("model_key",)),
    ]
    identities = []
    for identity in identity_candidates:
        if identity is not None and identity not in identities:
            identities.append(identity)
    expected_identity = EXPECTED_MODEL_IDENTITIES[directory]
    model_id, model_id_unique = _one_saved(sources, ("model_id", "checkpoint"))
    requested_revision, requested_unique = _one_saved(
        sources, ("requested_revision", "revision")
    )
    resolved_revision, resolved_unique = _one_saved(
        sources, ("resolved_revision", "resolved_model_revision", "resolved_commit_hash")
    )
    fingerprint, fingerprint_unique = _one_saved(
        sources,
        (
            "local_checkpoint_sha256",
            "checkpoint_fingerprint",
            "checkpoint_sha256",
            "fingerprint",
        ),
    )
    processor_identity, _ = _one_saved(
        sources,
        (
            "processor_identity",
            "processor_name_or_path",
            "feature_extractor_identity",
            "tokenizer_identity",
        ),
    )
    config_identity, _ = _one_saved(
        sources,
        ("config_identity", "model_config_sha256", "config_sha256"),
    )
    layers = layer_name_sets[0]
    reasons: list[dict[str, str]] = []
    if identities != [expected_identity]:
        reasons.append(
            _reason(
                "FAIL",
                "model_identity_mismatch",
                f"expected {expected_identity!r}; saved identities {identities!r}",
            )
        )
    if registered_hf_model_id is None:
        reasons.append(
            _reason(
                "WARN",
                "registered_hf_model_id_unresolved",
                "model registry was unavailable or lacked this model identity",
            )
        )
    if not model_id_unique:
        reasons.append(_reason("FAIL", "checkpoint_id_unresolved", repr(model_id)))
    if not requested_unique and not resolved_unique and not fingerprint_unique:
        reasons.append(
            _reason(
                "FAIL",
                "revision_or_fingerprint_unresolved",
                "no unique saved revision or checkpoint fingerprint",
            )
        )
    if processor_identity is None:
        reasons.append(
            _reason(
                "WARN",
                "processor_identity_unresolved",
                "saved extraction metadata has no processor identity",
            )
        )
    if config_identity is None:
        reasons.append(
            _reason(
                "WARN",
                "config_identity_unresolved",
                "saved extraction metadata has no immutable processor/model config identity",
            )
        )
    extracted_ids = extraction["recording_id"].astype(str).tolist()
    if sorted(extracted_ids) != sorted(recording_ids) or len(extracted_ids) != len(
        set(extracted_ids)
    ):
        reasons.append(
            _reason(
                "FAIL",
                "extraction_manifest_recordings_mismatch",
                f"expected {recording_ids!r}; observed {extracted_ids!r}",
            )
        )
    if any(names != layers for names in layer_name_sets[1:]):
        reasons.append(
            _reason("FAIL", "layer_definitions_mismatch", "recordings differ")
        )

    sample_rate, _ = _preferred_saved(
        sources, ("input_sample_rate_hz", "sample_rate_hz", "processed_sample_rate")
    )
    native_rate, _ = _preferred_saved(
        sources, ("native_frame_rate_hz", "native_rate_hz_from_runtime")
    )
    canonical_rate, _ = _preferred_saved(
        sources, ("canonical_rate_hz", "canonical_frame_rate_hz")
    )
    chunk_length, _ = _preferred_saved(
        sources,
        (
            "batch_seconds_effective",
            "batch_seconds_requested",
            "batch_seconds",
            "chunk_length_seconds",
        ),
    )
    overlap, _ = _preferred_saved(
        sources,
        (
            "overlap_seconds_effective",
            "overlap_seconds_requested",
            "chunk_overlap_seconds",
            "overlap_seconds",
        ),
    )
    retained_rule, _ = _preferred_saved(
        sources, ("stitching", "retained_frame_rule", "edge_retention_rule")
    )
    layer_definitions = _saved_values(sources, ("layers",))
    packages = run_metadata.get("package_versions", run_metadata.get("packages"))
    git = run_metadata.get("git", {"commit": run_metadata.get("git_commit")})
    provenance = {
        "model_directory": directory,
        "expected_model_identity": expected_identity,
        "saved_model_identities": identities,
        "checkpoint_id": model_id,
        "registered_hf_model_id": registered_hf_model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "checkpoint_fingerprint": fingerprint,
        "processor_identity": processor_identity,
        "config_identity": config_identity,
        "sample_rate_hz": sample_rate,
        "native_frame_rate_hz": native_rate,
        "canonical_frame_rate_hz": canonical_rate,
        "chunk_length_seconds": chunk_length,
        "chunk_overlap_seconds": overlap,
        "retained_frame_rule": retained_rule,
        "layer_count": len(layers),
        "layer_names": layers,
        "saved_layer_definitions": layer_definitions,
        "package_versions": packages,
        "git": git,
        "status": _status(reasons),
        "reasons": reasons,
    }

    adapter, adapter_saved = _one_saved(sources, ("adapter",))
    bidirectional, bidirectional_saved = _one_saved(
        sources,
        ("bidirectional", "is_bidirectional", "bidirectional_context"),
        "unknown",
    )
    future_context, future_saved = _one_saved(
        sources,
        ("future_context", "uses_future_context", "future_context_seconds"),
        "unknown",
    )
    receptive_field, receptive_saved = _one_saved(
        sources,
        (
            "receptive_field_seconds",
            "receptive_field_samples",
            "receptive_field",
        ),
        "unknown",
    )
    native_resolution = {}
    for label, keys in (
        ("timing_rule", ("native_timing_rule",)),
        ("frame_rate_hz", ("native_frame_rate_hz", "native_rate_hz_from_runtime")),
        ("frame_stride_samples", ("frame_stride_samples",)),
        ("frame_center_offset_samples", ("frame_center_offset_samples",)),
    ):
        value, saved = _one_saved(sources, keys)
        if saved:
            native_resolution[label] = value
    native_resolution_saved = bool(native_resolution)
    temporal = {
        "model_directory": directory,
        "adapter": adapter if adapter_saved else "unknown",
        "bidirectional_context": bidirectional,
        "future_context": future_context,
        "context_audit_support": (
            "supported" if bidirectional_saved and future_saved else "unsupported"
        ),
        "receptive_field": receptive_field,
        "receptive_field_support": "saved" if receptive_saved else "unknown",
        "chunk_length_seconds": chunk_length if chunk_length is not None else "unknown",
        "chunk_overlap_seconds": overlap if overlap is not None else "unknown",
        "edge_retention_rule": retained_rule or "unknown",
        "native_resolution": native_resolution or "unknown",
        "native_resolution_support": (
            "saved" if native_resolution_saved else "unknown"
        ),
        "evidence_policy": "saved_metadata_only_no_architecture_inference",
        "lag_convention": LAG_CONVENTION,
    }
    temporal_reasons: list[dict[str, str]] = []
    if not (bidirectional_saved and future_saved):
        temporal_reasons.append(
            _reason(
                "WARN",
                "unsupported_context_audit",
                "bidirectional/future-context fields are not fully saved",
            )
        )
    if not receptive_saved:
        temporal_reasons.append(
            _reason(
                "WARN",
                "receptive_field_unknown",
                "no receptive-field value is saved",
            )
        )
    if not native_resolution_saved:
        temporal_reasons.append(
            _reason(
                "WARN",
                "native_resolution_unknown",
                "no native-resolution value is saved",
            )
        )
    temporal["status"] = _status(temporal_reasons)
    temporal["reasons"] = temporal_reasons
    return provenance, temporal


def _atomic_write_json(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    for column in frame:
        if any(isinstance(value, (dict, list)) for value in frame[column]):
            frame[column] = frame[column].map(_json_cell)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _runtime_provenance(repository_root: Path) -> dict[str, Any]:
    packages = {}
    for name in ("speech-strf", "numpy", "pandas", "h5py", "soundfile"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "package_versions": packages}


def run_stage4_qc(
    manifest_path: str | Path = "outputs/manifest.csv",
    model_root: str | Path = "outputs",
    output_dir: str | Path = "outputs/stage4_revision/provenance_qc",
    *,
    endpoint_tolerance_seconds: float = 0.03,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate atomic machine-readable Stage 4 QC tables from saved sources."""
    if endpoint_tolerance_seconds < 0:
        raise ValueError("endpoint_tolerance_seconds must be nonnegative")
    manifest_path, model_root = Path(manifest_path), Path(model_root)
    output_dir = Path(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Required Stage 4 manifest unavailable: {manifest_path}"
        )
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception as exc:
        raise ValueError(f"Invalid Stage 4 manifest {manifest_path}: {exc}") from exc
    required = {"recording_id", "audio_path", "alignment_path"}
    missing_columns = sorted(required - set(manifest))
    if missing_columns:
        raise ValueError(
            f"Invalid Stage 4 manifest {manifest_path}: missing columns {missing_columns}"
        )
    if manifest.empty or manifest["recording_id"].isna().any():
        raise ValueError(f"Invalid Stage 4 manifest {manifest_path}: no valid recordings")
    recording_ids = manifest["recording_id"].astype(str).tolist()
    if len(recording_ids) != len(set(recording_ids)):
        raise ValueError(
            f"Invalid Stage 4 manifest {manifest_path}: duplicate recording_id values"
        )

    recordings = []
    for row in manifest.to_dict("records"):
        recording_id = str(row["recording_id"])
        if pd.isna(row["audio_path"]) or not str(row["audio_path"]).strip():
            raise FileNotFoundError(
                f"Audio source unavailable for {recording_id}: manifest audio_path is empty"
            )
        if pd.isna(row["alignment_path"]) or not str(row["alignment_path"]).strip():
            raise FileNotFoundError(
                f"Alignment source unavailable for {recording_id}: "
                "manifest alignment_path is empty"
            )
        recordings.append(
            _annotation_qc(
                recording_id,
                Path(str(row["audio_path"])),
                Path(str(row["alignment_path"])),
                endpoint_tolerance_seconds,
            )
        )

    registry: dict[str, Any] = {}
    if model_config_path is not None:
        model_config_path = Path(model_config_path)
        try:
            registry_config = yaml.safe_load(
                model_config_path.read_text(encoding="utf-8")
            )
            registry = registry_config["models"]
        except Exception as exc:
            raise ValueError(
                f"Invalid model registry {model_config_path}: {exc}"
            ) from exc

    models, temporal = [], []
    for directory in STAGE4_MODEL_DIRECTORIES:
        identity = EXPECTED_MODEL_IDENTITIES[directory]
        provenance, context = _model_qc(
            directory,
            model_root / directory,
            recording_ids,
            registry.get(identity, {}).get("model_id"),
        )
        models.append(provenance)
        temporal.append(context)

    totals = {
        "recording_count": len(recordings),
        "phone_feature_family": "phonetic",
        "phonetic_family_definition": (
            "forced-aligned phone onsets and phone-category indicators, "
            "including tone-bearing labels where present"
        ),
        "phone_count": sum(row["phone_count"] for row in recordings),
        "distinct_phone_category_count": len(
            {
                label
                for row in recordings
                for label in row["phone_category_distribution"]
            }
        ),
        "phone_category_distribution": dict(
            sorted(
                sum(
                    (
                        Counter(row["phone_category_distribution"])
                        for row in recordings
                    ),
                    Counter(),
                ).items()
            )
        ),
        "word_count": sum(row["word_count"] for row in recordings),
        "empty_label_count": sum(row["empty_label_count"] for row in recordings),
        "out_of_bounds_count": sum(
            row["out_of_bounds_count"] for row in recordings
        ),
        "tolerated_numerical_out_of_bounds_count": sum(
            row["tolerated_numerical_out_of_bounds_count"] for row in recordings
        ),
        "tolerated_empty_out_of_bounds_count": sum(
            row["tolerated_empty_out_of_bounds_count"] for row in recordings
        ),
        "fatal_out_of_bounds_count": sum(
            row["fatal_out_of_bounds_count"] for row in recordings
        ),
        "overlapping_count": sum(row["overlapping_count"] for row in recordings),
        "reversed_count": sum(row["reversed_count"] for row in recordings),
        "max_annotation_audio_endpoint_mismatch_seconds": max(
            (
                row["annotation_audio_endpoint_mismatch_seconds"]
                for row in recordings
            ),
            default=0.0,
        ),
    }
    all_statuses = [row["status"] for row in recordings + models + temporal]
    overall = max(all_statuses, key=STATUS_ORDER.get) if all_statuses else "FAIL"
    overall_reasons = [
        {
            "scope": row.get("recording_id", row.get("model_directory")),
            **reason,
        }
        for row in recordings + models + temporal
        for reason in row["reasons"]
    ]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "reasons": overall_reasons,
        "manifest_path": str(manifest_path),
        "model_root": str(model_root),
        "model_directories": list(STAGE4_MODEL_DIRECTORIES),
        "lag_convention": LAG_CONVENTION,
        "runtime_provenance": _runtime_provenance(manifest_path.resolve().parent.parent),
        "annotation_totals": totals,
        "recordings": recordings,
        "models": models,
        "temporal_context": temporal,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "provenance_qc.json", report)
    _atomic_write_csv(output_dir / "provenance_qc.csv", models)
    _atomic_write_csv(output_dir / "recording_annotation_qc.csv", recordings)
    _atomic_write_csv(output_dir / "temporal_context.csv", temporal)
    _atomic_write_csv(output_dir / "annotation_totals.csv", [totals])
    return report


generate_stage4_qc = run_stage4_qc

