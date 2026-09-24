#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from speech_strf.model_registry import RegistryEntry, get_model_entry
from speech_strf.pipeline import (
    extract_model,
    figure_model,
    finalize_metadata,
    fit_model,
    model_output_dir,
    output_identity_mismatches,
)
from speech_strf.provenance import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--analysis-config", default="configs/analysis.yaml")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--validation-report", default="outputs/validation_report.json")
    parser.add_argument("--features", default="outputs/features")
    parser.add_argument(
        "--stage",
        choices=["extract", "fit", "figures", "all"],
        default="all",
    )
    parser.add_argument("--recording-id", action="append")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    entry = get_model_entry(args.model_config, args.model)
    if not entry.enabled:
        raise SystemExit(f"{entry.key} is disabled: {entry.limitations}")
    if args.checkpoint:
        entry = RegistryEntry(entry.key, {**entry.values, "model_id": args.checkpoint})
    output = Path(args.output) if args.output else model_output_dir(entry)
    stages = {"extract", "fit", "figures"} if args.stage == "all" else {args.stage}
    if args.recording_id and stages != {"extract"}:
        raise SystemExit(
            "--recording-id is an extraction smoke-test option; use --stage extract"
        )
    prior_metadata_path = output / "run_metadata.json"
    if prior_metadata_path.exists() and "extract" not in stages:
        prior = json.loads(prior_metadata_path.read_text())
        mismatches = output_identity_mismatches(
            entry,
            prior,
            args.feature_config,
            args.analysis_config,
        )
        if mismatches:
            raise SystemExit(f"Existing output identity mismatch: {mismatches}")
    activation_store = output / "activations.h5"
    if "fit" in stages and "extract" not in stages and not activation_store.is_file():
        raise SystemExit(
            f"Missing activation store: {activation_store}. "
            "Run this model and output with --stage extract first."
        )
    kernels = None
    if "extract" in stages:
        activation_store = extract_model(
            entry,
            args.manifest,
            args.validation_report,
            recording_ids=args.recording_id,
            allow_download=args.allow_download,
            overwrite=args.overwrite,
            output_dir=output,
        )
    results_path = output / "fit" / "all_layers_results.csv"
    if "fit" in stages:
        results_path, kernels = fit_model(
            entry,
            activation_store,
            args.features,
            args.manifest,
            args.analysis_config,
            args.feature_config,
            output,
        )
    if "figures" in stages:
        if not results_path.exists():
            raise SystemExit(f"Missing combined fit results: {results_path}")
        if kernels is None:
            layer_dirs = sorted(
                path for path in (output / "fit").iterdir() if path.is_dir()
            )
            if not layer_dirs:
                raise SystemExit("No per-layer kernel directories found")
            kernel_file = np.load(layer_dirs[0] / "kernels.npz")
            kernels = {
                layer_dirs[0].name: {
                    key: kernel_file[key] for key in kernel_file.files
                }
            }
        figure_model(entry, results_path, kernels, output)
    analysis = load_config(args.analysis_config)
    finalize_metadata(
        entry,
        output,
        args.manifest,
        args.model_config,
        args.feature_config,
        args.analysis_config,
        int(analysis["random_seed"]),
    )
    print(f"{entry.key} {args.stage} complete: {output}")


if __name__ == "__main__":
    main()

