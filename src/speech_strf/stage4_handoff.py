"""Build a compact, auditable manuscript handoff from Stage 4 outputs."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .stage4_compute_scope import (
    ALL_SCOPE_MODELS,
    CONTROL_MODELS,
    NULL_MODELS,
)
from .stage4_encoding import FAMILIES
from .stage4_nulls import SUBSTANTIVE_FAMILIES
from .stage4_runner import Stage4Runner, sha256_file, validate_unit
from .stage4_statistics import (
    benjamini_hochberg,
    exact_paired_sign_flip,
    recording_cluster_bootstrap_ci,
)


HANDOFF_SCHEMA_VERSION = 1
TABLE_DIR = "tables"
VERIFY_DIR = "verification"
PROVENANCE_DIR = "provenance_qc"
PREDICTION_DIR = "prediction_verification"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame, columns: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _status_warning_rows(value: Any, source: str, location: str = "root") -> list[dict]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        severity = str(value.get("status", value.get("severity", ""))).upper()
        if severity in {"WARN", "FAIL", "ERROR"}:
            rows.append(
                {
                    "source": source,
                    "location": location,
                    "severity": severity,
                    "reason": value.get("reason", value.get("message", "")),
                    "context_json": json.dumps(value, sort_keys=True, default=_json_default),
                }
            )
        for key, child in value.items():
            rows.extend(
                _status_warning_rows(child, source, f"{location}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(
                _status_warning_rows(child, source, f"{location}[{index}]")
            )
    return rows


def _infer_recording_effects(
    frame: pd.DataFrame,
    expected_recordings: Sequence[str],
    *,
    group_columns: Sequence[str] = ("model", "layer"),
    family_order: Sequence[str] = FAMILIES,
    effect_column: str = "delta_r2",
    bootstrap_samples: int = 10_000,
    seed: int = 17,
    estimand: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Infer within each model/layer using recordings, never folds, as units."""
    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(), incomplete
    expected = [str(value) for value in expected_recordings]
    group_keys = [*group_columns]
    for keys, layer_frame in frame.groupby(group_keys, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_keys, keys))
        layer_rows: list[dict[str, Any]] = []
        for family in family_order:
            selected = layer_frame[layer_frame["family"].astype(str).eq(family)]
            observed_ids = selected["recording_id"].astype(str).tolist()
            if (
                len(selected) != len(expected)
                or set(observed_ids) != set(expected)
                or len(observed_ids) != len(set(observed_ids))
            ):
                incomplete.append(
                    {
                        **identity,
                        "family": family,
                        "reason": "recording_set_incomplete",
                        "expected_recordings": len(expected),
                        "observed_recordings": len(set(observed_ids)),
                    }
                )
                continue
            indexed = selected.assign(
                recording_id=selected["recording_id"].astype(str)
            ).set_index("recording_id")
            values = indexed.loc[expected, effect_column].to_numpy(dtype=float)
            durations = indexed.loc[expected, "duration_seconds"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.isfinite(durations).all():
                incomplete.append(
                    {
                        **identity,
                        "family": family,
                        "reason": "nonfinite_effect_or_duration",
                    }
                )
                continue
            test = exact_paired_sign_flip(values, estimand=estimand)
            ci = recording_cluster_bootstrap_ci(
                values,
                n_bootstrap=bootstrap_samples,
                seed=seed,
            )
            weighted_ci = recording_cluster_bootstrap_ci(
                values,
                weights=durations,
                n_bootstrap=bootstrap_samples,
                seed=seed,
            )
            layer_rows.append(
                {
                    **identity,
                    "family": family,
                    "equal_recording_mean_delta_r2": test["effect"],
                    "duration_weighted_mean_delta_r2": float(
                        np.average(values, weights=durations)
                    ),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "duration_weighted_ci_low": weighted_ci[0],
                    "duration_weighted_ci_high": weighted_ci[1],
                    "p_value": test["p_value"],
                    "positive_effect_recordings": test["positive_count"],
                    "negative_effect_recordings": test["negative_count"],
                    "zero_effect_recordings": test["zero_count"],
                    "recording_count": len(expected),
                    "inferential_unit": "recording",
                    "estimand": estimand,
                }
            )
        if len(layer_rows) == len(family_order):
            q_values = benjamini_hochberg(
                [float(value["p_value"]) for value in layer_rows]
            )
            for row, q_value in zip(layer_rows, q_values):
                row["q_value"] = q_value
                row["survives_bh_fdr_0_05"] = bool(q_value < 0.05)
        else:
            for row in layer_rows:
                row["q_value"] = np.nan
                row["survives_bh_fdr_0_05"] = False
                row["bh_status"] = "not_applied_incomplete_family_set"
        rows.extend(layer_rows)
    return pd.DataFrame(rows), incomplete


def _paired_control_comparison(
    control: pd.DataFrame,
    original: pd.DataFrame,
    *,
    label: str,
    expected_recordings: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    keys = ["model", "layer", "recording_id", "family"]
    if control.empty or original.empty:
        return pd.DataFrame(), pd.DataFrame(), [
            {"comparison": label, "reason": "control_or_original_missing"}
        ]
    left = control[keys + ["delta_r2", "duration_seconds"]].rename(
        columns={"delta_r2": "control_delta_r2"}
    )
    right = original[keys + ["delta_r2"]].rename(
        columns={"delta_r2": "original_delta_r2"}
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    paired["control_minus_original_delta_r2"] = (
        paired["control_delta_r2"] - paired["original_delta_r2"]
    )
    paired["comparison"] = label
    inference, incomplete = _infer_recording_effects(
        paired,
        expected_recordings,
        effect_column="control_minus_original_delta_r2",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        estimand=f"equal-recording mean {label} minus original conditional delta R2",
    )
    return paired, inference, incomplete


def _unit_scores(
    unit: Path,
    *,
    split_kind: str,
    analysis_role: str,
    missing: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    valid, reason = validate_unit(unit)
    if not valid:
        missing.append({"unit": str(unit), "reason": reason})
        return pd.DataFrame(), [], []
    scores = pd.read_csv(unit / "scores.csv")
    selected = scores[scores["split_kind"].astype(str).eq(split_kind)].copy()
    if selected.empty:
        missing.append(
            {
                "unit": str(unit),
                "reason": f"split_kind_missing:{split_kind}",
            }
        )
        return pd.DataFrame(), [], []
    selected["analysis_role"] = analysis_role
    selected["unit_path"] = str(unit)
    status = json.loads((unit / "status.json").read_text(encoding="utf-8"))
    selected["alpha_policy"] = status.get(
        "hyperparameter_policy",
        "nested training-only alpha selection",
    )
    pca = json.loads((unit / "pca_reports.json").read_text(encoding="utf-8"))
    capacity = json.loads(
        (unit / "capacity_reports.json").read_text(encoding="utf-8")
    )
    return selected, pca, capacity


def _pca_tables(reports: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = pd.DataFrame(reports)
    if folds.empty:
        return folds, pd.DataFrame()
    if "cumulative_explained_variance_ratio" in folds:
        folds["cumulative_explained_variance_ratio"] = folds[
            "cumulative_explained_variance_ratio"
        ].map(json.dumps)
    summary = (
        folds.groupby(["model", "layer", "variant"], as_index=False)
        .agg(
            original_target_dimensionality=("target_units_before_pca", "first"),
            retained_components_min=("achieved_components", "min"),
            retained_components_max=("achieved_components", "max"),
            coverage_mean=("total_explained_variance_ratio", "mean"),
            coverage_sd=("total_explained_variance_ratio", "std"),
            coverage_min=("total_explained_variance_ratio", "min"),
            coverage_max=("total_explained_variance_ratio", "max"),
            fold_count=("outer_fold", "nunique"),
        )
    )
    return folds, summary


def _flatten_config(value: Any, prefix: str = "") -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_config(child, path)
    elif isinstance(value, list):
        yield {"config_path": prefix, "value": json.dumps(value)}
    else:
        yield {"config_path": prefix, "value": value}


def _collect_prediction_inventory(
    output_root: Path,
) -> tuple[pd.DataFrame, Path | None]:
    rows = []
    candidates = []
    for path in sorted(output_root.rglob("predictions.npz")):
        if "handoff" in path.parts or ".corrupt-" in str(path):
            continue
        unit = path.parent
        valid, reason = validate_unit(unit)
        status_path = unit / "status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {}
        )
        actual_hash = sha256_file(path)
        rows.append(
            {
                "model": status.get("model"),
                "variant": status.get("variant"),
                "layer": status.get("layer"),
                "unit_path": str(unit),
                "prediction_path": str(path),
                "sha256": actual_hash,
                "bytes": path.stat().st_size,
                "unit_integrity_valid": valid,
                "unit_integrity_reason": reason,
                "manifest_hash_matches": (
                    status.get("artifacts", {})
                    .get("predictions.npz", {})
                    .get("sha256")
                    == actual_hash
                ),
            }
        )
        if valid:
            candidates.append(path)
    representative = min(candidates, key=lambda value: (value.stat().st_size, str(value))) if candidates else None
    return pd.DataFrame(rows), representative


def _copy_source_snapshot(
    repository_root: Path,
    destination: Path,
    *,
    patch_base: str | None,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    patterns = (
        "configs/*.yaml",
        "configs/stage4_models.tsv",
        "src/speech_strf/stage4_*.py",
        "scripts/*stage4*.py",
        "slurm/stage4_*.sbatch",
        "STAGE4_*.md",
    )
    copied = []
    for pattern in patterns:
        for source in sorted(repository_root.glob(pattern)):
            if not source.is_file():
                continue
            target = destination / source.relative_to(repository_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(source.relative_to(repository_root)))

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return ""

    metadata = {
        "commit": git("rev-parse", "HEAD").strip() or None,
        "branch": git("branch", "--show-current").strip() or None,
        "status_porcelain": git("status", "--porcelain").splitlines(),
        "copied_source_files": copied,
        "patch_base": patch_base,
    }
    if patch_base:
        patch = git("diff", "--binary", f"{patch_base}..HEAD", "--", ".")
        (destination / "repository.patch").write_text(patch, encoding="utf-8")
    _write_json(destination / "repository_metadata.json", metadata)
    return metadata


def _slurm_summary(log_root: Path, job_ids: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = {str(value) for value in job_ids if str(value)}
    for path in log_root.glob("*"):
        ids.update(re.findall(r"(?<!\d)(\d{4,})(?!\d)", path.name))
    columns = (
        "JobIDRaw,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocCPUS,NodeList"
    )
    jobs = pd.DataFrame()
    if ids:
        try:
            process = subprocess.run(
                [
                    "sacct",
                    "-j",
                    ",".join(sorted(ids)),
                    "--parsable2",
                    "--noheader",
                    f"--format={columns}",
                ],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            process = subprocess.CompletedProcess(
                args=["sacct"],
                returncode=127,
                stdout="",
                stderr=str(exc),
            )
        if process.returncode == 0 and process.stdout.strip():
            rows = list(
                csv.DictReader(
                    process.stdout.splitlines(),
                    delimiter="|",
                    fieldnames=columns.split(","),
                )
            )
            jobs = pd.DataFrame(rows)
        elif process.returncode != 0:
            jobs = pd.DataFrame(
                [{"State": "SACCT_UNAVAILABLE", "ExitCode": process.stderr.strip()}]
            )
    logs = pd.DataFrame(
        [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(log_root.glob("*"))
            if path.is_file()
        ]
    )
    return jobs, logs


def _write_sha256sums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_stage4_handoff(root: str | Path) -> dict[str, Any]:
    """Verify hashes and enforce the compact-data denylist."""
    root = Path(root)
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError("SHA256SUMS is missing")
    checked = 0
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        listed.add(relative)
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Handoff hash mismatch: {relative}")
        checked += 1
    delivered = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != delivered:
        raise ValueError(
            "SHA256SUMS coverage mismatch; "
            f"missing={sorted(delivered - listed)}, extra={sorted(listed - delivered)}"
        )
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name == "activations.h5"
            or "features_rich" in path.parts
            or "__incomplete__" in path.name
            or path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"}
            or (
                path.suffix.lower() == ".npz"
                and path.name != "predictions.npz"
            )
        )
    ]
    predictions = list(root.rglob("predictions.npz"))
    if forbidden:
        raise ValueError(f"Forbidden large inputs in handoff: {forbidden}")
    if len(predictions) > 1 or any(
        PREDICTION_DIR not in path.parts for path in predictions
    ):
        raise ValueError("Handoff may contain exactly one representative prediction")
    return {
        "state": "valid",
        "files_checked": checked,
        "representative_prediction_count": len(predictions),
    }


def build_stage4_handoff(
    config_path: str | Path = "configs/stage4_revision.yaml",
    *,
    output: str | Path | None = None,
    job_ids: Sequence[str] = (),
    patch_base: str | None = None,
    archive_existing: bool = False,
) -> dict[str, Any]:
    """Build and atomically publish the Stage 4 manuscript handoff."""
    runner = Stage4Runner(config_path)
    output_root = runner.output_root
    destination = (
        runner.resolve(output)
        if output is not None
        else output_root / "handoff"
    )
    if destination.exists():
        if not archive_existing:
            raise FileExistsError(f"Refusing to overwrite handoff {destination}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        os.replace(
            destination,
            destination.with_name(f"{destination.name}.previous-{stamp}"),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".handoff.tmp-", dir=destination.parent)
    )
    missing: list[dict[str, Any]] = []
    incomplete_inference: list[dict[str, Any]] = []
    expected_recordings = runner._recording_ids()
    bootstrap = int(runner.config["statistics"]["bootstrap_samples"])
    seed = int(runner.config["random_seed"])
    manifest_root = (
        output_root / "manifests" / "deadline_compute_scope"
    )
    resolved_path = manifest_root / "resolved_model_layers.json"
    resolved = (
        json.loads(resolved_path.read_text(encoding="utf-8"))
        if resolved_path.is_file()
        else {}
    )

    try:
        table_root = temporary / TABLE_DIR
        primary_frames = []
        loro_frames = []
        pca_reports: list[dict[str, Any]] = []
        capacity_reports: list[dict[str, Any]] = []

        hubert_units = (
            resolved.get("hubert_base_original", {}).get("units", {})
        )
        for layer, unit_path in sorted(hubert_units.items()):
            grouped, pca, capacity = _unit_scores(
                Path(unit_path),
                split_kind="sensitivity",
                analysis_role="primary_grouped_hubert_base",
                missing=missing,
            )
            loro, _, _ = _unit_scores(
                Path(unit_path),
                split_kind="primary",
                analysis_role="hubert_base_loro_sensitivity",
                missing=missing,
            )
            primary_frames.append(grouped)
            loro_frames.append(loro)
            pca_reports.extend(
                value for value in pca if value.get("split_kind") == "sensitivity"
            )
            capacity_reports.extend(capacity)

        primary_tasks = manifest_root / "primary_fast_refits.tsv"
        if primary_tasks.is_file():
            for task in _read_tsv(primary_tasks):
                for layer in task["layers"].split(","):
                    unit = output_root / "primary_fast_refits" / task["model"] / layer
                    scores, pca, capacity = _unit_scores(
                        unit,
                        split_kind="primary_fast_grouped",
                        analysis_role="primary_grouped_fixed_alpha_refit",
                        missing=missing,
                    )
                    primary_frames.append(scores)
                    pca_reports.extend(pca)
                    capacity_reports.extend(capacity)
        else:
            missing.append(
                {"unit": str(primary_tasks), "reason": "primary_task_manifest_missing"}
            )
        primary = (
            pd.concat(primary_frames, ignore_index=True)
            if any(not value.empty for value in primary_frames)
            else pd.DataFrame()
        )
        loro = (
            pd.concat(loro_frames, ignore_index=True)
            if any(not value.empty for value in loro_frames)
            else pd.DataFrame()
        )
        _write_csv(table_root / "primary_recording_level.csv", primary)
        primary_inference, incomplete = _infer_recording_effects(
            primary,
            expected_recordings,
            bootstrap_samples=bootstrap,
            seed=seed,
            estimand="equal-recording mean primary grouped conditional delta R2",
        )
        incomplete_inference.extend(incomplete)
        _write_csv(table_root / "primary_inference.csv", primary_inference)

        # HuBERT split sensitivity: direct paired recording table and conclusions.
        _write_csv(table_root / "hubert_base_loro_recording_level.csv", loro)
        hubert_grouped = primary[primary.get("model", pd.Series(dtype=str)).eq("hubert_base")]
        loro_inference, incomplete = _infer_recording_effects(
            loro,
            expected_recordings,
            bootstrap_samples=bootstrap,
            seed=seed,
            estimand="equal-recording mean HuBERT Base LORO conditional delta R2",
        )
        incomplete_inference.extend(incomplete)
        _write_csv(table_root / "hubert_base_loro_inference.csv", loro_inference)
        split_paired, split_inference, split_missing = _paired_control_comparison(
            loro,
            hubert_grouped,
            label="loro_minus_grouped",
            expected_recordings=expected_recordings,
            bootstrap_samples=bootstrap,
            seed=seed,
        )
        incomplete_inference.extend(split_missing)
        _write_csv(table_root / "hubert_base_split_comparison_recording.csv", split_paired)
        _write_csv(table_root / "hubert_base_split_comparison_inference.csv", split_inference)
        grouped_inference = primary_inference[
            primary_inference.get("model", pd.Series(dtype=str)).eq("hubert_base")
        ]
        if not loro_inference.empty and not grouped_inference.empty:
            columns = ["model", "layer", "family"]
            split_conclusions = loro_inference[
                columns
                + [
                    "equal_recording_mean_delta_r2",
                    "q_value",
                    "survives_bh_fdr_0_05",
                ]
            ].merge(
                grouped_inference[
                    columns
                    + [
                        "equal_recording_mean_delta_r2",
                        "q_value",
                        "survives_bh_fdr_0_05",
                    ]
                ],
                on=columns,
                suffixes=("_loro", "_grouped"),
                validate="one_to_one",
            )
            split_conclusions["direction_changed"] = (
                np.sign(split_conclusions["equal_recording_mean_delta_r2_loro"])
                != np.sign(split_conclusions["equal_recording_mean_delta_r2_grouped"])
            )
            split_conclusions["fdr_conclusion_changed"] = (
                split_conclusions["survives_bh_fdr_0_05_loro"]
                != split_conclusions["survives_bh_fdr_0_05_grouped"]
            )
            split_conclusions["conclusion_changed"] = (
                split_conclusions["direction_changed"]
                | split_conclusions["fdr_conclusion_changed"]
            )
        else:
            split_conclusions = pd.DataFrame()
        _write_csv(
            table_root / "hubert_base_split_conclusion_changes.csv",
            split_conclusions,
        )

        controls_by_variant: dict[str, list[pd.DataFrame]] = {
            "rich": [],
            "capacity": [],
            "zero_lag": [],
        }
        controls_manifest = manifest_root / "selected_depth_controls.tsv"
        if controls_manifest.is_file():
            for task in _read_tsv(controls_manifest):
                for depth in ("input", "middle", "final"):
                    layer = task[depth]
                    variant = task["variant"]
                    subdir = runner.config["variants"][variant]["output_subdir"]
                    unit = output_root / subdir / task["model"] / layer
                    scores, pca, capacity = _unit_scores(
                        unit,
                        split_kind="primary",
                        analysis_role=f"{variant}_{depth}_depth_control",
                        missing=missing,
                    )
                    if not scores.empty:
                        scores["depth"] = depth
                    controls_by_variant[variant].append(scores)
                    pca_reports.extend(pca)
                    capacity_reports.extend(capacity)
        else:
            missing.append(
                {"unit": str(controls_manifest), "reason": "control_manifest_missing"}
            )

        control_outputs: dict[str, pd.DataFrame] = {}
        control_inferences: dict[str, pd.DataFrame] = {}
        for variant, frames in controls_by_variant.items():
            frame = (
                pd.concat(frames, ignore_index=True)
                if any(not value.empty for value in frames)
                else pd.DataFrame()
            )
            control_outputs[variant] = frame
            _write_csv(table_root / f"{variant}_recording_level.csv", frame)
            inference, incomplete = _infer_recording_effects(
                frame,
                expected_recordings,
                bootstrap_samples=bootstrap,
                seed=seed,
                estimand=f"equal-recording mean {variant} conditional delta R2",
            )
            incomplete_inference.extend(incomplete)
            control_inferences[variant] = inference
            _write_csv(table_root / f"{variant}_inference.csv", inference)
            paired, contrast, contrast_missing = _paired_control_comparison(
                frame,
                primary,
                label=f"{variant}_minus_original",
                expected_recordings=expected_recordings,
                bootstrap_samples=bootstrap,
                seed=seed,
            )
            incomplete_inference.extend(contrast_missing)
            _write_csv(table_root / f"{variant}_vs_original_recording.csv", paired)
            _write_csv(table_root / f"{variant}_vs_original_inference.csv", contrast)

        rich_config = yaml.safe_load(
            (runner.root / "configs" / "features_stage4_rich.yaml").read_text(
                encoding="utf-8"
            )
        )
        _write_csv(
            table_root / "rich_acoustic_feature_inventory.csv",
            pd.DataFrame(_flatten_config(rich_config)),
        )

        capacity_flat = []
        for report in capacity_reports:
            for family, values in report.get("families", {}).items():
                capacity_flat.append(
                    {
                        "model": report.get("model"),
                        "layer": report.get("layer"),
                        "variant": report.get("variant"),
                        "outer_fold": report.get("outer_fold"),
                        "family": family,
                        "achieved_rank": report.get("rank"),
                        "input_columns": values.get("input_columns"),
                        "output_columns": values.get("output_columns"),
                        "predictor_explained_variance_coverage": values.get(
                            "explained_variance_coverage"
                        ),
                    }
                )
        _write_csv(
            table_root / "capacity_predictor_coverage.csv",
            pd.DataFrame(capacity_flat),
        )

        # Family ordering comparisons use inferential equal-recording effects.
        ordering_rows = []
        for variant in ("rich", "capacity"):
            control_inference = control_inferences[variant]
            if control_inference.empty or primary_inference.empty:
                continue
            for (model, layer), group in control_inference.groupby(["model", "layer"]):
                baseline = primary_inference[
                    primary_inference["model"].eq(model)
                    & primary_inference["layer"].eq(layer)
                ]
                if len(group) != len(FAMILIES) or len(baseline) != len(FAMILIES):
                    continue
                control_order = group.sort_values(
                    "equal_recording_mean_delta_r2", ascending=False
                )["family"].tolist()
                original_order = baseline.sort_values(
                    "equal_recording_mean_delta_r2", ascending=False
                )["family"].tolist()
                ordering_rows.append(
                    {
                        "variant": variant,
                        "model": model,
                        "layer": layer,
                        "control_family_order": ">".join(control_order),
                        "original_family_order": ">".join(original_order),
                        "ordering_identical": control_order == original_order,
                    }
                )
        _write_csv(
            table_root / "control_family_ordering.csv",
            pd.DataFrame(ordering_rows),
        )

        zero_inference = control_inferences["zero_lag"]
        if not zero_inference.empty and not primary_inference.empty:
            keys = ["model", "layer", "family"]
            zero_stability = zero_inference[
                keys
                + [
                    "equal_recording_mean_delta_r2",
                    "q_value",
                    "survives_bh_fdr_0_05",
                ]
            ].merge(
                primary_inference[
                    keys
                    + [
                        "equal_recording_mean_delta_r2",
                        "q_value",
                        "survives_bh_fdr_0_05",
                    ]
                ],
                on=keys,
                suffixes=("_zero_lag", "_five_lag"),
                validate="one_to_one",
            )
            zero_stability["direction_stable"] = (
                np.sign(zero_stability["equal_recording_mean_delta_r2_zero_lag"])
                == np.sign(zero_stability["equal_recording_mean_delta_r2_five_lag"])
            )
            zero_stability["statistical_conclusion_stable"] = (
                zero_stability["survives_bh_fdr_0_05_zero_lag"]
                == zero_stability["survives_bh_fdr_0_05_five_lag"]
            )
            zero_stability["directionally_and_statistically_stable"] = (
                zero_stability["direction_stable"]
                & zero_stability["statistical_conclusion_stable"]
            )
        else:
            zero_stability = pd.DataFrame()
        _write_csv(
            table_root / "zero_lag_conclusion_stability.csv",
            zero_stability,
        )

        pca_folds, pca_summary = _pca_tables(pca_reports)
        _write_csv(table_root / "pca_coverage_by_fold.csv", pca_folds)
        _write_csv(table_root / "pca_coverage_summary.csv", pca_summary)

        # Fixed-alpha structured null inventory, offsets, and observed-minus-null.
        null_unit_rows = []
        null_score_frames = []
        shift_rows = []
        null_manifest_path = manifest_root / "structured_nulls.tsv"
        if null_manifest_path.is_file():
            for task in _read_tsv(null_manifest_path):
                unit = (
                    output_root
                    / "null_controls"
                    / "fixed_alpha_20"
                    / task["model"]
                    / task["layer"]
                    / f"null_{int(task['shift']):03d}"
                )
                valid, reason = validate_unit(unit)
                null_unit_rows.append(
                    {
                        **task,
                        "unit_path": str(unit),
                        "complete": valid,
                        "reason": reason,
                    }
                )
                if not valid:
                    missing.append({"unit": str(unit), "reason": reason})
                    continue
                scores = pd.read_csv(unit / "scores.csv")
                scores["null_shift"] = int(task["shift"])
                null_score_frames.append(scores)
                status = json.loads(
                    (unit / "status.json").read_text(encoding="utf-8")
                )
                for recording_id, recording in status.get(
                    "null_shift_manifest", {}
                ).get("recordings", {}).items():
                    for family, shift in recording.get("families", {}).items():
                        shift_rows.append(
                            {
                                "model": task["model"],
                                "layer": task["layer"],
                                "null_shift": int(task["shift"]),
                                "recording_id": recording_id,
                                "family": family,
                                **shift,
                            }
                        )
        else:
            missing.append(
                {"unit": str(null_manifest_path), "reason": "null_manifest_missing"}
            )
        null_units = pd.DataFrame(null_unit_rows)
        null_scores = (
            pd.concat(null_score_frames, ignore_index=True)
            if null_score_frames
            else pd.DataFrame()
        )
        _write_csv(table_root / "structured_null_unit_status.csv", null_units)
        _write_csv(table_root / "structured_null_shift_offsets.csv", pd.DataFrame(shift_rows))
        _write_csv(table_root / "structured_null_recording_results.csv", null_scores)

        null_paired = pd.DataFrame()
        null_inference = pd.DataFrame()
        if not null_scores.empty and not primary.empty:
            null_mean = (
                null_scores.groupby(
                    ["model", "layer", "recording_id", "family"],
                    as_index=False,
                )
                .agg(
                    mean_null_delta_r2=("delta_r2", "mean"),
                    completed_null_shifts=("null_shift", "nunique"),
                )
            )
            observed = primary[
                primary["family"].isin(SUBSTANTIVE_FAMILIES)
            ][
                [
                    "model",
                    "layer",
                    "recording_id",
                    "family",
                    "delta_r2",
                    "duration_seconds",
                ]
            ].rename(columns={"delta_r2": "observed_delta_r2"})
            null_paired = observed.merge(
                null_mean,
                on=["model", "layer", "recording_id", "family"],
                how="inner",
                validate="one_to_one",
            )
            null_paired["observed_minus_mean_null_delta_r2"] = (
                null_paired["observed_delta_r2"]
                - null_paired["mean_null_delta_r2"]
            )
            null_inference, incomplete = _infer_recording_effects(
                null_paired,
                expected_recordings,
                family_order=SUBSTANTIVE_FAMILIES,
                effect_column="observed_minus_mean_null_delta_r2",
                bootstrap_samples=bootstrap,
                seed=seed,
                estimand=(
                    "equal-recording mean observed minus mean fixed-alpha "
                    "structured-shift-null conditional delta R2"
                ),
            )
            incomplete_inference.extend(incomplete)
        _write_csv(table_root / "structured_null_observed_minus_null.csv", null_paired)
        _write_csv(table_root / "structured_null_inference.csv", null_inference)

        # Provenance/QC products are copied, not merely pointed to.
        provenance_target = temporary / PROVENANCE_DIR
        provenance_target.mkdir(parents=True, exist_ok=True)
        warning_rows = []
        for source in (
            output_root / "audit" / "input_audit.json",
            output_root / "provenance_qc" / "provenance_qc.json",
        ):
            if source.is_file():
                value = json.loads(source.read_text(encoding="utf-8"))
                warning_rows.extend(_status_warning_rows(value, str(source)))
        for source in sorted((output_root / "provenance_qc").glob("*")):
            if source.is_file():
                shutil.copy2(source, provenance_target / source.name)
        audit_source = output_root / "audit" / "input_audit.json"
        if audit_source.is_file():
            shutil.copy2(audit_source, provenance_target / audit_source.name)
        model_metadata_root = provenance_target / "model_metadata"
        for model in ALL_SCOPE_MODELS:
            source_root = runner._model(model)
            target_root = model_metadata_root / model
            for name in (
                "run_metadata.json",
                "layer_metadata.json",
                "comparability_contract.json",
                "extraction_manifest.csv",
            ):
                source = source_root / name
                if source.is_file():
                    target_root.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target_root / name)
                else:
                    missing.append(
                        {
                            "unit": str(source),
                            "reason": "model_provenance_file_missing",
                        }
                    )

        verification = temporary / VERIFY_DIR
        source_metadata = _copy_source_snapshot(
            runner.root,
            verification / "source",
            patch_base=patch_base,
        )
        manifest_target = verification / "resolved_manifests"
        if manifest_root.is_dir():
            shutil.copytree(manifest_root, manifest_target)
        governance_target = verification / "governance"
        for name, configured_path in runner.config.get(
            "governance_inputs", {}
        ).items():
            source = runner.resolve(configured_path)
            if source.is_file():
                governance_target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, governance_target / f"{name}{source.suffix}")
            else:
                missing.append(
                    {
                        "unit": str(source),
                        "reason": "governance_file_missing",
                    }
                )

        logs_root = output_root / "logs"
        jobs, log_inventory = _slurm_summary(logs_root, job_ids)
        _write_csv(verification / "slurm_job_completion.csv", jobs)
        _write_csv(verification / "slurm_log_inventory.csv", log_inventory)
        test_target = verification / "test_reports"
        test_target.mkdir(parents=True, exist_ok=True)
        for source in sorted(logs_root.glob("*.xml")):
            shutil.copy2(source, test_target / source.name)

        resource_rows = []
        for status_path in sorted(output_root.rglob("status.json")):
            if "handoff" in status_path.parts or ".corrupt-" in str(status_path):
                continue
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if "fit_runtime_seconds" in status or "peak_rss_kib" in status:
                resource_rows.append(
                    {
                        "status_path": str(status_path),
                        "model": status.get("model"),
                        "layer": status.get("layer"),
                        "variant": status.get("variant"),
                        "null_index": status.get("null_index"),
                        "state": status.get("state"),
                        "runtime_seconds": status.get("fit_runtime_seconds"),
                        "peak_rss_kib": status.get("peak_rss_kib"),
                    }
                )
        _write_csv(
            verification / "runtime_peak_memory.csv",
            pd.DataFrame(resource_rows),
        )

        error_pattern = re.compile(
            r"(warning|error|traceback|failed|cancelled|out.of.memory)",
            re.IGNORECASE,
        )
        for source in sorted(logs_root.glob("*")):
            if not source.is_file() or source.suffix not in {".out", ".err", ".log"}:
                continue
            try:
                for line_number, line in enumerate(
                    source.read_text(encoding="utf-8", errors="replace").splitlines(),
                    1,
                ):
                    if error_pattern.search(line):
                        warning_rows.append(
                            {
                                "source": str(source),
                                "location": f"line:{line_number}",
                                "severity": "LOG",
                                "reason": line[:1000],
                                "context_json": "",
                            }
                        )
            except OSError:
                continue
        _write_csv(
            verification / "warnings_errors.csv",
            pd.DataFrame(warning_rows),
        )

        prediction_inventory, representative = _collect_prediction_inventory(
            output_root
        )
        prediction_root = temporary / PREDICTION_DIR
        _write_csv(prediction_root / "prediction_inventory.csv", prediction_inventory)
        representative_info = None
        if representative is not None:
            copied = prediction_root / "predictions.npz"
            shutil.copy2(representative, copied)
            representative_status = representative.parent / "status.json"
            if representative_status.is_file():
                shutil.copy2(
                    representative_status,
                    prediction_root / "source_unit_status.json",
                )
            representative_info = {
                "source": str(representative),
                "delivered": str(copied.relative_to(temporary)),
                "sha256": sha256_file(copied),
                "bytes": copied.stat().st_size,
                "selection_rule": "smallest integrity-valid observed prediction archive",
            }
            _write_json(
                prediction_root / "representative_prediction.json",
                representative_info,
            )

        if not list(test_target.glob("*")):
            missing.append(
                {
                    "unit": str(logs_root / "*.xml"),
                    "reason": "test_report_missing",
                }
            )
        if jobs.empty:
            missing.append(
                {
                    "unit": "sacct",
                    "reason": "slurm_job_completion_summary_empty",
                }
            )
        missing_frame = pd.DataFrame(missing)
        _write_csv(verification / "missing_incomplete_units.csv", missing_frame)
        _write_csv(
            verification / "incomplete_inference_cells.csv",
            pd.DataFrame(incomplete_inference),
        )
        test_reports = list(test_target.glob("*"))
        qc_failures = [
            value for value in warning_rows if value.get("severity") in {"FAIL", "ERROR"}
        ]
        package_status = "COMPLETE"
        if missing or incomplete_inference or qc_failures:
            package_status = "PARTIAL"
        report = {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "state": package_status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "compact auditable Stage 4 manuscript-results handoff",
            "primary_outputs_deleted_note": (
                "Missing primary fast-refit units are expected when manually "
                "removed from the server; no fold-level substitute is fabricated."
            ),
            "expected_models": list(ALL_SCOPE_MODELS),
            "expected_recordings": expected_recordings,
            "missing_unit_count": len(missing),
            "incomplete_inference_cell_count": len(incomplete_inference),
            "qc_failure_count": len(qc_failures),
            "prediction_inventory_count": len(prediction_inventory),
            "representative_prediction": representative_info,
            "test_report_count": len(test_reports),
            "source": source_metadata,
            "large_data_excluded": [
                "activation stores",
                "raw audio",
                "feature NPZ archives",
                "all but one prediction NPZ",
            ],
            "sha256_scope": "all delivered files except SHA256SUMS itself",
        }
        _write_json(temporary / "HANDOFF.json", report)
        (temporary / "README.md").write_text(
            "# Stage 4 manuscript handoff\n\n"
            f"Package status: **{package_status}**.\n\n"
            "Use `HANDOFF.json` and `verification/missing_incomplete_units.csv` "
            "before making claims. Tables contain recording-level estimands; "
            "outer folds are not inferential units. `SHA256SUMS` covers every "
            "delivered file except itself. Activation stores, raw audio, feature "
            "archives, and bulk predictions are intentionally excluded.\n",
            encoding="utf-8",
        )
        _write_sha256sums(temporary)
        verification_result = verify_stage4_handoff(temporary)
        report["verification"] = verification_result
        # Rewriting HANDOFF changes its hash, so regenerate the sums once.
        _write_json(temporary / "HANDOFF.json", report)
        _write_sha256sums(temporary)
        verify_stage4_handoff(temporary)
        os.replace(temporary, destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
