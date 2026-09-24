#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from speech_strf.comparability import (
    compare_contracts,
    compare_hubert_regression,
    write_json,
)
from speech_strf.model_registry import get_model_entry
from speech_strf.pipeline import model_output_dir


def _store_ids(path: Path) -> list[str]:
    with h5py.File(path) as store:
        return sorted(name for name in store if not name.startswith("__"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config", default="configs/models.yaml")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--features", default="outputs/features")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--analysis-config", default="configs/analysis.yaml")
    parser.add_argument("--candidate-root")
    parser.add_argument(
        "--reference-root",
        default="outputs",
        help="Locked legacy HuBERT root containing activations.h5 and fit outputs",
    )
    parser.add_argument("--run-hubert-regression", action="store_true")
    args = parser.parse_args()

    entry = get_model_entry(args.model_config, args.model)
    candidate_root = (
        Path(args.candidate_root) if args.candidate_root else model_output_dir(entry)
    )
    reference_root = Path(args.reference_root)
    candidate_contract_path = candidate_root / "comparability_contract.json"
    if not candidate_contract_path.exists():
        raise SystemExit(f"Missing candidate contract: {candidate_contract_path}")
    candidate_contract = json.loads(candidate_contract_path.read_text())
    reference_contract_path = reference_root / "hubert_reference_contract.json"
    if not reference_contract_path.exists():
        raise SystemExit(
            f"Missing frozen HuBERT contract: {reference_contract_path}. Run "
            "scripts/freeze_hubert_reference.py against the completed reference first."
        )
    reference_contract = json.loads(reference_contract_path.read_text())
    report = compare_contracts(reference_contract, candidate_contract)
    report.update(
        {
            "model_key": entry.key,
            "reference_model": "hubert_large_reference",
            "reference_contract_source": str(reference_contract_path),
        }
    )
    candidate_store = candidate_root / "activations.h5"
    reference_store = reference_root / "activations.h5"
    if candidate_store.exists() and reference_store.exists():
        candidate_ids = _store_ids(candidate_store)
        reference_ids = _store_ids(reference_store)
        report["activation_stimulus_ids"] = {
            "match": candidate_ids == reference_ids,
            "reference": reference_ids,
            "candidate": candidate_ids,
        }
        report["comparable"] &= candidate_ids == reference_ids
    else:
        report["activation_stimulus_ids"] = {
            "match": False,
            "error": "Reference or candidate activation store is missing",
        }
        report["comparable"] = False
    if args.run_hubert_regression:
        reference_results = reference_root / "fit" / "all_layers_results.csv"
        if not reference_results.exists():
            reference_results = reference_root / "fit" / "results.csv"
        regression = compare_hubert_regression(
            reference_store,
            candidate_store,
            reference_results,
            candidate_root / "fit" / "all_layers_results.csv",
        )
        report["hubert_refactor_regression"] = regression
        report["comparable"] &= regression["passed"]
    destination = write_json(report, candidate_root / "comparability_report.json")
    print(f"Comparable to HuBERT reference: {report['comparable']}")
    print(destination)
    raise SystemExit(0 if report["comparable"] else 2)


if __name__ == "__main__":
    main()

