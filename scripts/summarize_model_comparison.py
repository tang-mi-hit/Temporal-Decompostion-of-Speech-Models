#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from speech_strf.model_registry import load_model_registry
from speech_strf.model_summary import (
    save_predictability_summary,
    summarize_predictability,
)
from speech_strf.provenance import write_run_manifest


def _root_overrides(values: list[str] | None) -> dict[str, Path]:
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--model-root must use MODEL_KEY=PATH")
        key, path = value.split("=", 1)
        overrides[key] = Path(path)
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="Candidate model key; repeatable")
    parser.add_argument("--model-root", action="append", help="MODEL_KEY=PATH override")
    parser.add_argument("--model-config", default="configs/models.yaml")
    parser.add_argument(
        "--reference-results", default="outputs/fit/all_layers_results.csv"
    )
    parser.add_argument(
        "--reference-contract", default="outputs/hubert_reference_contract.json"
    )
    parser.add_argument("--output", default="outputs/model_comparison")
    args = parser.parse_args()

    _, entries = load_model_registry(args.model_config)
    reference_key = "hubert_large_reference"
    if not Path(args.reference_contract).exists():
        raise SystemExit(
            f"Missing {args.reference_contract}; run freeze_hubert_reference.py first"
        )
    reference_contract = json.loads(Path(args.reference_contract).read_text())
    if reference_contract.get("locked_reference", {}).get("model_key") != reference_key:
        raise SystemExit("Reference contract does not identify locked HuBERT Large")
    if not Path(args.reference_results).exists():
        raise SystemExit(f"Missing HuBERT results: {args.reference_results}")

    overrides = _root_overrides(args.model_root)
    selected = args.model
    if selected is None:
        selected = [
            key
            for key, entry in entries.items()
            if key != reference_key
            and entry.enabled
            and (
                overrides.get(key, Path(f"outputs/{key}"))
                / "fit"
                / "all_layers_results.csv"
            ).exists()
        ]
    if not selected:
        raise SystemExit("No completed candidate models selected or discovered")

    results = {
        reference_key: pd.read_csv(args.reference_results),
    }
    metadata = {
        reference_key: {
            "model_id": entries[reference_key].model_id,
            "input_modality": entries[reference_key].input_modality,
        }
    }
    sources = {
        f"{reference_key}_results": args.reference_results,
        "hubert_reference_contract": args.reference_contract,
    }
    for key in selected:
        if key not in entries:
            raise SystemExit(f"Unknown model key: {key}")
        root = overrides.get(key, Path(f"outputs/{key}"))
        report_path = root / "comparability_report.json"
        results_path = root / "fit" / "all_layers_results.csv"
        if not report_path.exists():
            raise SystemExit(
                f"{key} has no comparability report; run "
                f"compare_to_hubert_reference.py --model {key}"
            )
        report = json.loads(report_path.read_text())
        if not report.get("comparable", False):
            raise SystemExit(f"{key} failed comparability checks: {report_path}")
        if not results_path.exists():
            raise SystemExit(f"{key} results are missing: {results_path}")
        results[key] = pd.read_csv(results_path)
        metadata[key] = {
            "model_id": entries[key].model_id,
            "input_modality": entries[key].input_modality,
        }
        sources[f"{key}_results"] = results_path
        sources[f"{key}_comparability"] = report_path

    model_table, layer_table, family_table, fold_table = summarize_predictability(
        results, metadata, reference_key
    )
    save_predictability_summary(
        args.output,
        model_table,
        layer_table,
        family_table,
        fold_table,
        sources,
    )
    write_run_manifest(
        args.model_config,
        args.output,
        extra={
            "reference_model": reference_key,
            "candidate_models": selected,
            "summary_type": "descriptive_representation_predictability",
        },
    )
    print(model_table.to_string(index=False))
    print(f"Wrote cross-model summary: {args.output}")


if __name__ == "__main__":
    main()

