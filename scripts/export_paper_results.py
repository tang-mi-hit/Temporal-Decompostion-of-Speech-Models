#!/usr/bin/env python3
"""Export or verify the paper's recording-level replication tables."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from speech_strf.stage4_handoff import (
    _infer_recording_effects,
    build_stage4_handoff,
    verify_stage4_handoff,
)
from speech_strf.stage4_encoding import FAMILIES
from speech_strf.stage4_nulls import SUBSTANTIVE_FAMILIES
from speech_strf.stage4_runner import Stage4Runner


def export_aggregate_results(runner, package):
    tables = package / "tables"
    destination = runner.output_root / "paper_summary"
    destination.mkdir(parents=True, exist_ok=True)
    expected = runner._recording_ids()
    cases = [
        ("hubert_base_grouped", "primary_recording_level", "delta_r2", 1, 13, FAMILIES),
        ("hubert_base_loro", "hubert_base_loro_recording_level", "delta_r2", 1, 13, FAMILIES),
        ("rich", "rich_recording_level", "delta_r2", 6, 3, SUBSTANTIVE_FAMILIES),
        ("capacity", "capacity_recording_level", "delta_r2", 6, 3, SUBSTANTIVE_FAMILIES),
        ("zero_lag", "zero_lag_recording_level", "delta_r2", 9, 3, FAMILIES),
        ("structured_null", "structured_null_observed_minus_null",
         "observed_minus_mean_null_delta_r2", 4, 1, SUBSTANTIVE_FAMILIES),
    ]
    summaries = []
    for label, filename, effect, models, layers, families in cases:
        frame = pd.read_csv(tables / f"{filename}.csv")
        if label.startswith("hubert_base"):
            frame = frame.loc[frame["model"].eq("hubert_base")]
        frame = frame.loc[frame["family"].isin(families)]
        counts = frame.groupby(["model", "recording_id", "family"])["layer"].nunique()
        if (
            frame["model"].nunique() != models
            or len(frame) != models * layers * len(expected) * len(families)
            or len(counts) != models * len(expected) * len(families)
            or not counts.eq(layers).all()
            or not set(frame["recording_id"]) == set(expected)
        ):
            raise ValueError(f"Incomplete recording/layer/model panel for {label}")
        if label == "structured_null" and not frame["completed_null_shifts"].eq(20).all():
            raise ValueError("The circular-shift comparison requires all 20 offsets")
        columns = [effect, "duration_seconds"]
        per_model = frame.groupby(["model", "recording_id", "family"])[columns].mean()
        pooled = per_model.groupby(["recording_id", "family"]).mean().reset_index()
        pooled["analysis"] = label
        summary, incomplete = _infer_recording_effects(
            pooled,
            expected,
            group_columns=("analysis",),
            family_order=families,
            effect_column=effect,
            bootstrap_samples=int(runner.config["statistics"]["bootstrap_samples"]),
            seed=int(runner.config["random_seed"]),
            estimand="equal-recording mean after equal layer and checkpoint averaging",
        )
        if incomplete:
            raise ValueError(f"Incomplete recording vectors for {label}: {incomplete}")
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(
        destination / "aggregate_inference.csv", index=False
    )

    fold_tables = []
    for model in runner.model_names:
        source = runner._model(model) / "fit" / "all_layers_results.csv"
        folds = pd.read_csv(source)
        folds = folds.loc[folds["feature_family"].isin(FAMILIES)]
        fold_tables.append(
            folds.groupby(["outer_fold", "feature_family"], as_index=False)
            ["conditional_delta_r2"].mean().assign(model=model)
        )
    combined = pd.concat(fold_tables, ignore_index=True)
    keys = ["outer_fold", "feature_family"]
    full = combined.groupby(keys)["conditional_delta_r2"].mean()
    omissions = []
    for model in runner.model_names:
        omitted = combined.loc[combined["model"].ne(model)]
        means = omitted.groupby(keys)["conditional_delta_r2"].mean()
        omissions.append(
            pd.DataFrame({
                "full_panel_delta_r2": full,
                "omission_delta_r2": means,
                "relative_change_percent": 100 * ((means - full) / full.where(full != 0)),
            }).reset_index().assign(omitted_checkpoint=model)
        )
    pd.concat(omissions, ignore_index=True).to_csv(
        destination / "checkpoint_omissions_by_fold.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_analysis.yaml")
    parser.add_argument(
        "--output",
        default="outputs/stage4_revision/handoff",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Timestamp-archive an existing handoff before rebuilding",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing --output instead of building",
    )
    args = parser.parse_args()
    if args.verify:
        result = verify_stage4_handoff(args.output)
    else:
        runner = Stage4Runner(args.config)
        result = build_stage4_handoff(
            args.config,
            output=args.output,
            archive_existing=args.archive_existing,
            local_run=True,
        )
        export_aggregate_results(runner, runner.resolve(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
