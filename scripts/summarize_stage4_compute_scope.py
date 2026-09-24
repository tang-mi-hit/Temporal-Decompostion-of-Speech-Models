#!/usr/bin/env python3
"""Summarize deadline-scope primary scores with recording-level weights."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from speech_strf.stage4_compute_scope import PRIMARY_FAST_MODELS
from speech_strf.stage4_runner import Stage4Runner, discover_layers, validate_unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_revision.yaml")
    parser.add_argument(
        "--output",
        default="outputs/stage4_revision/compute_scope_summaries",
    )
    args = parser.parse_args()
    runner = Stage4Runner(args.config)
    frames = []
    sensitivity = []
    pca_rows = []

    for model in ("hubert_base", *PRIMARY_FAST_MODELS):
        layers = discover_layers(runner._model(model) / "activations.h5")
        for layer in layers:
            if model == "hubert_base":
                unit = runner._destination(model, layer, variant="original")
            else:
                unit = runner._fast_refit_destination(model, layer)
            valid, reason = validate_unit(unit)
            if not valid:
                raise SystemExit(f"Incomplete primary unit {unit}: {reason}")
            scores = pd.read_csv(unit / "scores.csv")
            reports = json.loads(
                (unit / "pca_reports.json").read_text(encoding="utf-8")
            )
            if model == "hubert_base":
                grouped = scores.query("split_kind == 'sensitivity'").copy()
                loro = scores.query("split_kind == 'primary'").copy()
                grouped["analysis_role"] = "primary_grouped_fixed_original"
                loro["analysis_role"] = "hubert_base_loro_sensitivity"
                frames.append(grouped)
                sensitivity.append(loro)
                pca_rows.extend(
                    value for value in reports
                    if value.get("split_kind") == "sensitivity"
                )
            else:
                selected = scores.query(
                    "split_kind == 'primary_fast_grouped'"
                ).copy()
                selected["analysis_role"] = "primary_grouped_fixed_alpha_refit"
                frames.append(selected)
                pca_rows.extend(
                    value for value in reports
                    if value.get("split_kind") == "primary_fast_grouped"
                )
    recording = pd.concat(frames, ignore_index=True)
    if recording.groupby(["model", "layer", "family"])["recording_id"].nunique().min() != 12:
        raise SystemExit("Primary summary lacks 12 separately scored recordings")

    summary_rows = []
    for keys, group in recording.groupby(["model", "layer", "family"], sort=True):
        weights = group["duration_seconds"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model": keys[0],
                "layer": keys[1],
                "family": keys[2],
                "recording_count": group["recording_id"].nunique(),
                "equal_recording_mean_full_r2": group["full_r2"].mean(),
                "duration_weighted_mean_full_r2": np.average(
                    group["full_r2"], weights=weights
                ),
                "equal_recording_mean_delta_r2": group["delta_r2"].mean(),
                "duration_weighted_mean_delta_r2": np.average(
                    group["delta_r2"], weights=weights
                ),
                "inferential_unit": "recording",
            }
        )
    destination = runner.resolve(args.output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite summary {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        recording.to_csv(temporary / "primary_recording_level_results.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(
            temporary / "primary_equal_and_duration_weighted.csv", index=False
        )
        pd.concat(sensitivity, ignore_index=True).to_csv(
            temporary / "hubert_base_loro_sensitivity.csv", index=False
        )
        pca = pd.DataFrame(pca_rows)
        if not pca.empty:
            pca["cumulative_explained_variance_ratio"] = pca[
                "cumulative_explained_variance_ratio"
            ].map(json.dumps)
        pca.to_csv(temporary / "primary_pca_coverage_by_fold.csv", index=False)
        report = {
            "state": "complete",
            "primary_models": ["hubert_base", *PRIMARY_FAST_MODELS],
            "primary_estimand": (
                "recording-level full R2 and conditional delta R2 from original "
                "five grouped outer folds"
            ),
            "inferential_unit": "recording_not_outer_fold",
            "hubert_base_loro_role": "sensitivity_only",
            "recording_rows": len(recording),
            "summary_rows": len(summary_rows),
            "pca_coverage_rows": len(pca_rows),
        }
        (temporary / "summary_status.json").write_text(
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


if __name__ == "__main__":
    main()
