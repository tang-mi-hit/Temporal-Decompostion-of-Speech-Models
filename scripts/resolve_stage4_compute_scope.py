#!/usr/bin/env python3
"""Resolve and validate deadline-constrained Stage 4 model/layer manifests."""

from __future__ import annotations

import argparse
import json

from speech_strf.stage4_compute_scope import (
    PRIMARY_FAST_MODELS,
    load_legacy_fixed_alphas,
    publish_compute_scope_manifests,
    validate_legacy_alpha_compatibility,
)
from speech_strf.stage4_runner import Stage4Runner, discover_layers, validate_unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_revision.yaml")
    parser.add_argument(
        "--output",
        default="outputs/stage4_revision/manifests/deadline_compute_scope",
    )
    parser.add_argument(
        "--refresh-stale-manifest",
        action="store_true",
        help="Archive a differing existing manifest directory before publishing",
    )
    args = parser.parse_args()
    runner = Stage4Runner(args.config)
    runner._audit()
    model_layers = {
        model: discover_layers(runner._model(model) / "activations.h5")
        for model in runner.model_names
    }

    # Fail before submission unless every fixed-alpha source has the exact
    # original feature, grid, lag, PCA, split, and recording contract.
    alpha_sources = {}
    for model in PRIMARY_FAST_MODELS:
        contract, results = validate_legacy_alpha_compatibility(
            runner._model(model),
            manifest_path=runner.resolve(runner.config["manifest"]),
            features_dir=runner.resolve(runner.config["features"]["original"]),
            feature_config_path=runner.root / "configs" / "features.yaml",
            analysis_config_path=runner.root / "configs" / "analysis.yaml",
        )
        for layer in model_layers[model]:
            load_legacy_fixed_alphas(results, layer)
        alpha_sources[model] = {
            "comparability_contract": str(contract),
            "all_layer_results": str(results),
        }

    hubert_units = {}
    measured_resources = []
    for layer in model_layers["hubert_base"]:
        unit = runner._destination("hubert_base", layer, variant="original")
        valid, reason = validate_unit(unit)
        if not valid:
            raise SystemExit(
                f"Completed HuBERT Base original unit is not integrity-valid: "
                f"{layer}: {reason}"
            )
        hubert_units[layer] = str(unit)
        status = json.loads((unit / "status.json").read_text(encoding="utf-8"))
        measured_resources.append(
            {
                "layer": layer,
                "fit_runtime_seconds": status.get("fit_runtime_seconds"),
                "peak_rss_kib": status.get("peak_rss_kib"),
                "unit": str(unit),
            }
        )

    payload = publish_compute_scope_manifests(
        runner.resolve(args.output),
        model_layers=model_layers,
        hubert_original_units=hubert_units,
        fixed_alpha_sources=alpha_sources,
        measured_pilot_resources=measured_resources,
        preserve_stale=args.refresh_stale_manifest,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
