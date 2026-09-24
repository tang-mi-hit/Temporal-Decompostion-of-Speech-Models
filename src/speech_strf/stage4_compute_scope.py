"""Deadline-constrained Stage 4 task and fixed-alpha provenance contracts."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .comparability import COMPARABILITY_FIELDS, build_comparability_contract
from .stage4_encoding import FAMILIES


PRIMARY_FAST_MODELS = (
    "hubert_large_refactor_rerun",
    "wav2vec2_base",
    "wav2vec2_large",
    "wavlm_base_plus",
    "wavlm_large",
    "data2vec_audio_base",
    "xls_r_300m",
    "whisper_medium_encoder",
)
CONTROL_MODELS = (
    "hubert_base",
    "wav2vec2_large",
    "wavlm_large",
    "data2vec_audio_base",
    "xls_r_300m",
    "whisper_medium_encoder",
)
NULL_MODELS = (
    "hubert_base",
    "wav2vec2_large",
    "wavlm_large",
    "whisper_medium_encoder",
)
ALL_SCOPE_MODELS = ("hubert_base", *PRIMARY_FAST_MODELS)


def resolve_depth_layers(layers: Sequence[str]) -> dict[str, str]:
    """Resolve input/middle/final, choosing the lower layer on exact ties."""
    ordered = list(layers)
    if len(ordered) < 3 or len(ordered) != len(set(ordered)):
        raise ValueError("At least three unique ordered layers are required")
    input_candidates = [
        value for value in ordered if "input" in value.lower()
    ]
    if len(input_candidates) != 1:
        raise ValueError(
            "Layer manifest must identify exactly one input representation"
        )
    input_layer = input_candidates[0]
    encoder = [value for value in ordered if value != input_layer]
    depth = len(encoder)
    middle_index = min(
        range(depth),
        key=lambda index: (abs(2 * (index + 1) - depth), index),
    )
    return {
        "input": input_layer,
        "middle": encoder[middle_index],
        "final": encoder[-1],
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_legacy_alpha_compatibility(
    model_dir: str | Path,
    *,
    manifest_path: str | Path,
    features_dir: str | Path,
    feature_config_path: str | Path,
    analysis_config_path: str | Path,
) -> tuple[Path, Path]:
    """Require the alpha source to match current folds, inputs, lags, and PCA."""
    model_root = Path(model_dir)
    contract_path = model_root / "comparability_contract.json"
    results_path = model_root / "fit" / "all_layers_results.csv"
    if not contract_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            f"Missing original alpha source for {model_root}: "
            f"{contract_path}, {results_path}"
        )
    source = _load_json(contract_path)
    current = build_comparability_contract(
        manifest_path,
        features_dir,
        feature_config_path,
        analysis_config_path,
    )
    mismatches = [
        field
        for field in COMPARABILITY_FIELDS
        if source.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"{model_root.name}: original alpha source is incompatible for "
            f"{mismatches}"
        )
    return contract_path, results_path


def load_legacy_fixed_alphas(
    results_path: str | Path,
    layer: str,
    *,
    outer_folds: int = 5,
    reduced_families: Sequence[str] = FAMILIES,
) -> dict[int, dict[str, float]]:
    """Read one unique full/reduced alpha per legacy grouped outer fold."""
    frame = pd.read_csv(results_path)
    required_columns = {"layer", "outer_fold", "feature_family", "alpha"}
    if required_columns - set(frame):
        raise ValueError(
            f"Original results lack {sorted(required_columns - set(frame))}"
        )
    frame = frame.loc[frame["layer"].astype(str) == layer].copy()
    required_families = {"full", *map(str, reduced_families)}
    if set(frame["feature_family"].astype(str)) != required_families:
        raise ValueError(
            f"{layer}: alpha source must contain exactly {sorted(required_families)}"
        )
    result: dict[int, dict[str, float]] = {}
    for fold in range(outer_folds):
        subset = frame.loc[pd.to_numeric(frame["outer_fold"]) == fold]
        if len(subset) != len(required_families):
            raise ValueError(f"{layer}: fold {fold} alpha rows are missing or duplicated")
        values = {
            str(row.feature_family): float(row.alpha)
            for row in subset.itertuples(index=False)
        }
        if set(values) != required_families:
            raise ValueError(f"{layer}: fold {fold} alpha-family schema is invalid")
        result[fold] = values
    if len(frame) != outer_folds * len(required_families):
        raise ValueError(f"{layer}: unexpected alpha rows outside five folds")
    return result


def load_stage4_grouped_fixed_alphas(
    split_reports_path: str | Path,
    *,
    reduced_families: Sequence[str],
    split_kinds: Sequence[str] = ("sensitivity", "primary_fast_grouped"),
) -> dict[int, dict[str, float]]:
    """Read fixed alphas from completed Stage 4 grouped sensitivity reports."""
    reports = _load_json(Path(split_reports_path))
    selected = [
        value for value in reports if value.get("split_kind") in set(split_kinds)
    ]
    result: dict[int, dict[str, float]] = {}
    for report in selected:
        fold = int(report["outer_fold"])
        reduced = report["reduced_alphas"]
        result[fold] = {
            "full": float(report["full_alpha"]),
            **{
                family: float(reduced[family])
                for family in reduced_families
            },
        }
    if set(result) != set(range(5)):
        raise ValueError("Completed HuBERT Base grouped alpha source lacks five folds")
    return result


def _write_tsv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _selected_depth_control_rows(
    resolved: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in CONTROL_MODELS:
        for variant in ("rich", "capacity"):
            rows.append(
                {
                    "task_id": len(rows),
                    "model": model,
                    "variant": variant,
                    **resolved[model],
                }
            )
    for model in ALL_SCOPE_MODELS:
        rows.append(
            {
                "task_id": len(rows),
                "model": model,
                "variant": "zero_lag",
                **resolved[model],
            }
        )
    return rows


def _primary_fast_rows(
    model_layers: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": index,
            "model": model,
            "layers": ",".join(model_layers[model]),
            "layer_count": len(model_layers[model]),
        }
        for index, model in enumerate(PRIMARY_FAST_MODELS)
    ]


def _structured_null_rows(
    resolved: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": index,
            "model": model,
            "shift": shift,
            "layer": resolved[model]["middle"],
        }
        for index, (model, shift) in enumerate(
            (model, shift) for model in NULL_MODELS for shift in range(20)
        )
    ]


def _read_tsv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if b"\r" in raw or b"\x00" in raw or (raw and not raw.endswith(b"\n")):
        raise ValueError(
            f"{path.name} must use canonical LF line endings and contain no NUL bytes"
        )
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != list(fields):
            raise ValueError(f"{path.name} has an invalid schema")
        return list(reader)


def validate_compute_scope_manifests(
    manifest_dir: str | Path,
    *,
    model_layers: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Fail if persisted control tasks differ from current model layer schemas."""
    root = Path(manifest_dir)
    payload_path = root / "resolved_model_layers.json"
    task_paths = {
        "primary_fast": root / "primary_fast_refits.tsv",
        "controls": root / "selected_depth_controls.tsv",
        "nulls": root / "structured_nulls.tsv",
    }
    if not payload_path.is_file() or any(
        not path.is_file() for path in task_paths.values()
    ):
        raise FileNotFoundError(
            f"Compute-scope manifests are incomplete under {root}"
        )
    missing = set(ALL_SCOPE_MODELS) - set(model_layers)
    if missing:
        raise ValueError(f"Current layer schemas are missing models {sorted(missing)}")
    resolved = {
        model: resolve_depth_layers(model_layers[model])
        for model in sorted(model_layers)
    }
    saved = _load_json(payload_path)
    if saved.get("resolved_depth_layers") != resolved:
        raise ValueError(
            "Resolved layer manifest is stale relative to current activation stores"
        )
    definitions = {
        "primary_fast": (
            ("task_id", "model", "layers", "layer_count"),
            _primary_fast_rows(model_layers),
        ),
        "controls": (
            ("task_id", "model", "variant", "input", "middle", "final"),
            _selected_depth_control_rows(resolved),
        ),
        "nulls": (
            ("task_id", "model", "shift", "layer"),
            _structured_null_rows(resolved),
        ),
    }
    counts = {}
    for name, (fields, expected) in definitions.items():
        observed = _read_tsv(task_paths[name], fields)
        expected_text = [
            {key: str(value) for key, value in row.items()} for row in expected
        ]
        if observed != expected_text:
            raise ValueError(
                f"{task_paths[name].name} does not match current model-specific layers"
            )
        counts[name] = len(observed)
    return {
        "state": "valid",
        "model_count": len(model_layers),
        "task_counts": counts,
    }


validate_selected_depth_controls = validate_compute_scope_manifests


def publish_compute_scope_manifests(
    destination: str | Path,
    *,
    model_layers: Mapping[str, Sequence[str]],
    hubert_original_units: Mapping[str, str],
    fixed_alpha_sources: Mapping[str, Mapping[str, str]] | None = None,
    measured_pilot_resources: Sequence[Mapping[str, Any]] | None = None,
    preserve_stale: bool = False,
) -> dict[str, Any]:
    """Atomically publish deterministic deadline-scope manifests once."""
    root = Path(destination)
    missing = (
        set(PRIMARY_FAST_MODELS) | set(CONTROL_MODELS) | set(NULL_MODELS)
    ) - set(model_layers)
    if missing:
        raise ValueError(f"Cannot resolve layers for missing models {sorted(missing)}")
    resolved = {
        model: resolve_depth_layers(model_layers[model])
        for model in sorted(model_layers)
    }
    primary_rows = _primary_fast_rows(model_layers)
    control_rows = _selected_depth_control_rows(resolved)
    null_rows = _structured_null_rows(resolved)
    payload = {
        "schema_version": 1,
        "middle_rule": (
            "encoder layer minimizing abs((one_based_layer/depth)-0.5); "
            "lower layer wins exact ties"
        ),
        "primary_fast_models": list(PRIMARY_FAST_MODELS),
        "control_models": list(CONTROL_MODELS),
        "null_models": list(NULL_MODELS),
        "resolved_depth_layers": resolved,
        "hubert_base_original": {
            "action": "preserve_as_final_for_pilot_do_not_rerun",
            "units": dict(sorted(hubert_original_units.items())),
        },
        "fixed_alpha_sources": {
            model: dict(sorted(value.items()))
            for model, value in sorted((fixed_alpha_sources or {}).items())
        },
        "measured_hubert_base_original_resources": list(
            measured_pilot_resources or []
        ),
        "task_counts": {
            "primary_fast": len(primary_rows),
            "controls": len(control_rows),
            "nulls": len(null_rows),
        },
        "benchmark": {
            "model": PRIMARY_FAST_MODELS[0],
            "layers": [
                resolved[PRIMARY_FAST_MODELS[0]]["input"],
                resolved[PRIMARY_FAST_MODELS[0]]["middle"],
            ],
            "model_layer_count": len(model_layers[PRIMARY_FAST_MODELS[0]]),
            "candidate_workers": 2,
            "blas_threads_per_worker": 8,
        },
    }
    if root.exists():
        existing_path = root / "resolved_model_layers.json"
        if existing_path.is_file() and _load_json(existing_path) == payload:
            try:
                validate_compute_scope_manifests(
                    root,
                    model_layers=model_layers,
                )
            except (FileNotFoundError, ValueError):
                if not preserve_stale:
                    raise FileExistsError(
                        f"Refusing to overwrite invalid manifests at {root}"
                    )
            else:
                return payload
        if not preserve_stale:
            raise FileExistsError(
                f"Refusing to overwrite different resolved manifests at {root}"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = root.with_name(f"{root.name}.stale-{stamp}")
        os.replace(root, archived)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent)
    )
    try:
        (temporary / "resolved_model_layers.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_tsv(
            temporary / "primary_fast_refits.tsv",
            ("task_id", "model", "layers", "layer_count"),
            primary_rows,
        )
        _write_tsv(
            temporary / "selected_depth_controls.tsv",
            ("task_id", "model", "variant", "input", "middle", "final"),
            control_rows,
        )
        _write_tsv(
            temporary / "structured_nulls.tsv",
            ("task_id", "model", "shift", "layer"),
            null_rows,
        )
        os.replace(temporary, root)
    except BaseException:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    return payload
