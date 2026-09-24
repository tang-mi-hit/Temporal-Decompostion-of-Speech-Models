#!/usr/bin/env python3
"""Freeze a comparability contract beside, without modifying, legacy HuBERT outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd

from speech_strf.comparability import build_comparability_contract, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", default="outputs")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--features", default="outputs/features")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--analysis-config", default="configs/analysis.yaml")
    args = parser.parse_args()
    root = Path(args.reference_root)
    activation_store = root / "activations.h5"
    results_path = root / "fit" / "all_layers_results.csv"
    if not results_path.exists():
        raise SystemExit(
            f"Missing {results_path}; combine all locked HuBERT layer results first"
        )
    contract = build_comparability_contract(
        args.manifest, args.features, args.feature_config, args.analysis_config
    )
    results = pd.read_csv(results_path)
    if list(results.columns) != contract["result_schema"]:
        raise SystemExit(
            f"HuBERT result schema differs: {list(results.columns)} != "
            f"{contract['result_schema']}"
        )
    with h5py.File(activation_store) as store:
        ids = sorted(name for name in store if not name.startswith("__"))
        first = store[ids[0]]
        observed = json.loads(first.attrs.get("model_metadata_json", "{}"))
    if ids != contract["stimulus_ids"]:
        raise SystemExit(
            f"HuBERT activation IDs differ from manifest: {ids} != "
            f"{contract['stimulus_ids']}"
        )
    contract["locked_reference"] = {
        "model_key": "hubert_large_reference",
        "activation_store": str(activation_store),
        "results": str(results_path),
        "stimulus_ids": ids,
        "observed_model_metadata": observed,
    }
    destination = write_json(contract, root / "hubert_reference_contract.json")
    print(f"Locked HuBERT contract written without modifying reference arrays: {destination}")


if __name__ == "__main__":
    main()

