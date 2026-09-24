#!/usr/bin/env python3
"""Validate and consolidate the gated HuBERT Base Stage 4 pilot."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from speech_strf.stage4_encoding import FAMILIES
from speech_strf.stage4_nulls import SUBSTANTIVE_FAMILIES
from speech_strf.stage4_runner import Stage4Runner, discover_layers, validate_unit


def _collect_warnings(value: Any, location: str = "root") -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if str(value.get("status", "")).upper() == "WARN":
            warnings.append({"location": location, "value": value})
        for key, child in value.items():
            warnings.extend(_collect_warnings(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            warnings.extend(_collect_warnings(child, f"{location}[{index}]"))
    return warnings


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def validate_pilot(args: argparse.Namespace) -> dict[str, Any]:
    runner = Stage4Runner(args.config)
    model = args.model
    layers = discover_layers(runner._model(model) / "activations.h5")
    failures: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    resources: list[dict[str, Any]] = []
    recording_frames: list[pd.DataFrame] = []
    pca_rows: list[dict[str, Any]] = []
    null_frames: list[pd.DataFrame] = []
    null_manifests: list[dict[str, Any]] = []

    if len(layers) != 13:
        failures.append(
            {
                "unit": model,
                "reason": f"expected_13_layers_observed_{len(layers)}",
            }
        )

    expected_recordings = set(runner._recording_ids())
    for variant in ("original", "rich", "capacity", "zero_lag"):
        expected_splits = (
            {"primary", "sensitivity"} if variant == "original" else {"primary"}
        )
        for layer in layers:
            unit = runner._destination(model, layer, variant=variant)
            valid, reason = validate_unit(unit)
            if not valid:
                target = failures if unit.exists() else skipped
                target.append({"unit": str(unit), "reason": reason})
                continue
            status = json.loads((unit / "status.json").read_text(encoding="utf-8"))
            scores = pd.read_csv(unit / "scores.csv")
            observed_recordings = set(scores["recording_id"].astype(str))
            observed_families = set(scores["family"].astype(str))
            observed_splits = set(scores["split_kind"].astype(str))
            if (
                observed_recordings != expected_recordings
                or observed_families != set(FAMILIES)
                or observed_splits != expected_splits
            ):
                failures.append(
                    {"unit": str(unit), "reason": "pilot_score_scope_mismatch"}
                )
                continue
            recording_frames.append(scores)
            reports = json.loads(
                (unit / "pca_reports.json").read_text(encoding="utf-8")
            )
            primary_reports = [
                report for report in reports if report.get("split_kind") == "primary"
            ]
            if (
                len(primary_reports) != len(expected_recordings)
                or any(
                    "total_explained_variance_ratio" not in report
                    or "train_recording_ids" not in report
                    or "test_recording_ids" not in report
                    for report in reports
                )
            ):
                failures.append(
                    {"unit": str(unit), "reason": "pca_coverage_scope_mismatch"}
                )
                continue
            pca_rows.extend(reports)
            resources.append(
                {
                    "component": "observed_fit",
                    "variant": variant,
                    "model": model,
                    "layer": layer,
                    "null_index": None,
                    "runtime_seconds": status.get("fit_runtime_seconds"),
                    "peak_rss_kib": status.get("peak_rss_kib"),
                    "path": str(unit),
                }
            )

    for null_index in range(args.null_count):
        for layer in layers:
            unit = runner._destination(
                model,
                layer,
                variant="original",
                null_index=null_index,
                dry_run=True,
            )
            valid, reason = validate_unit(unit)
            if not valid:
                target = failures if unit.exists() else skipped
                target.append({"unit": str(unit), "reason": reason})
                continue
            status = json.loads((unit / "status.json").read_text(encoding="utf-8"))
            scores = pd.read_csv(unit / "scores.csv")
            if (
                set(scores["recording_id"].astype(str)) != expected_recordings
                or set(scores["family"].astype(str)) != set(SUBSTANTIVE_FAMILIES)
                or set(scores["split_kind"].astype(str)) != {"primary"}
            ):
                failures.append(
                    {"unit": str(unit), "reason": "null_score_scope_mismatch"}
                )
                continue
            scores["null_index"] = null_index
            null_frames.append(scores)
            null_manifests.append(
                {
                    "model": model,
                    "layer": layer,
                    **status["null_shift_manifest"],
                }
            )
            resources.append(
                {
                    "component": "null_fit",
                    "variant": "original",
                    "model": model,
                    "layer": layer,
                    "null_index": null_index,
                    "runtime_seconds": status.get("fit_runtime_seconds"),
                    "peak_rss_kib": status.get("peak_rss_kib"),
                    "path": str(unit),
                }
            )

    smoke_path = Path(args.smoke_status)
    try:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("state") != "complete" or smoke.get("null_shift_count") != 1:
            failures.append({"unit": str(smoke_path), "reason": "smoke_status_invalid"})
        resources.append(
            {
                "component": "functional_smoke",
                "variant": None,
                "model": model,
                "layer": smoke.get("layer"),
                "null_index": 0,
                "runtime_seconds": smoke.get("runtime_seconds"),
                "peak_rss_kib": smoke.get("peak_rss_kib"),
                "path": str(smoke_path.parent),
            }
        )
    except (OSError, ValueError) as exc:
        smoke = {}
        failures.append({"unit": str(smoke_path), "reason": f"smoke_unreadable:{exc}"})
    for label, path_value in (
        ("slurm_stdout", args.stdout),
        ("slurm_stderr", args.stderr),
        ("test_report", args.test_report),
    ):
        if not Path(path_value).is_file():
            failures.append({"unit": path_value, "reason": f"{label}_missing"})

    warnings: list[dict[str, Any]] = []
    for label, path in (
        ("audit", runner.audit_output),
        (
            "provenance_qc",
            runner.output_root / "provenance_qc" / "provenance_qc.json",
        ),
    ):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            warnings.extend(_collect_warnings(value, label))
            if str(value.get("status", "")).upper() == "FAIL":
                failures.append({"unit": str(path), "reason": f"{label}_failed"})
        except (OSError, ValueError) as exc:
            failures.append({"unit": str(path), "reason": f"{label}_unreadable:{exc}"})
    warnings.extend(
        {
            "location": "functional_smoke.unsupported_recordings_skipped",
            "value": value,
        }
        for value in smoke.get("unsupported_recordings_skipped", [])
    )

    report_name = f"pilot-{args.job_id}"
    destination = runner.output_root / "pilot_reports" / report_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite pilot report {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{report_name}.tmp-", dir=destination.parent)
    )
    try:
        recording_results = (
            pd.concat(recording_frames, ignore_index=True)
            if recording_frames
            else pd.DataFrame()
        )
        null_results = (
            pd.concat(null_frames, ignore_index=True)
            if null_frames
            else pd.DataFrame()
        )
        _write_table(recording_results, temporary / "recording_level_results.csv")
        _write_table(pd.DataFrame(pca_rows), temporary / "pca_coverage_by_fold.csv")
        _write_table(null_results, temporary / "null_results.csv")
        _write_table(pd.DataFrame(resources), temporary / "component_resources.csv")
        (temporary / "null_shift_manifests.json").write_text(
            json.dumps(null_manifests, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {
            "state": "complete" if not failures and not skipped else "failed",
            "kind": "hubert_base_checkpoint_pilot",
            "model": model,
            "layers": layers,
            "layer_count": len(layers),
            "variants": {
                "original": ["primary", "sensitivity"],
                "rich": ["primary"],
                "capacity": ["primary"],
                "zero_lag": ["primary"],
            },
            "null_scope": {
                "variant": "original",
                "split": "primary",
                "families": list(SUBSTANTIVE_FAMILIES),
                "null_count": args.null_count,
            },
            "slurm_stdout": args.stdout,
            "slurm_stderr": args.stderr,
            "test_report": args.test_report,
            "recording_level_rows": len(recording_results),
            "pca_coverage_rows": len(pca_rows),
            "null_result_rows": len(null_results),
            "null_manifest_count": len(null_manifests),
            "warnings": warnings,
            "failed_units": failures,
            "skipped_units": skipped,
            "artifacts": sorted(path.name for path in temporary.iterdir()),
        }
        (temporary / "integrity_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["state"] != "complete":
        raise SystemExit("HuBERT Base pilot integrity validation failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_revision.yaml")
    parser.add_argument("--model", default="hubert_base")
    parser.add_argument("--null-count", type=int, default=20)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--smoke-status", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--test-report", required=True)
    validate_pilot(parser.parse_args())


if __name__ == "__main__":
    main()
