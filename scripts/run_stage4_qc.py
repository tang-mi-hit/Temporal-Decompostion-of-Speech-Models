#!/usr/bin/env python3
"""Generate Stage 4 provenance, annotation, and temporal-context QC outputs."""

from __future__ import annotations

import argparse
import json
from typing import Any

from speech_strf.stage4_qc import run_stage4_qc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--model-root", default="outputs")
    parser.add_argument("--model-config", default="configs/models.yaml")
    parser.add_argument(
        "--output-dir", default="outputs/stage4_revision/provenance_qc"
    )
    parser.add_argument(
        "--endpoint-tolerance-seconds",
        type=float,
        default=0.03,
        help="Annotation/audio endpoint mismatch above this value is WARN",
    )
    return parser


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    report = run_stage4_qc(
        manifest_path=args.manifest,
        model_root=args.model_root,
        output_dir=args.output_dir,
        endpoint_tolerance_seconds=args.endpoint_tolerance_seconds,
        model_config_path=args.model_config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()

