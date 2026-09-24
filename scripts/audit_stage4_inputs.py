#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech_strf.stage4_audit import audit_stage4_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of the fixed nine-model Stage 4 inputs."
    )
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--features", default="outputs/features")
    parser.add_argument(
        "--model-root",
        default="outputs",
        help="Root containing the nine fixed model output directories.",
    )
    parser.add_argument(
        "--output",
        default="outputs/stage4_revision/audit/input_audit.json",
    )
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--data-config")
    parser.add_argument("--model-config")
    parser.add_argument("--feature-config")
    parser.add_argument("--analysis-config")
    parser.add_argument(
        "--revision-roadmap",
        default="../paper1_pipeline_20260914/stage3_review/revision_roadmap.json",
    )
    parser.add_argument(
        "--claim-surface-manifest",
        default=(
            "../paper1_pipeline_20260914/stage4_revision/"
            "claim_surface_manifest.json"
        ),
    )
    parser.add_argument("--timestamp-tolerance-seconds", type=float, default=1e-8)
    parser.add_argument("--expected-recording-count", type=int, default=12)
    args = parser.parse_args()

    config_root = Path(args.config_root)
    config_paths = {
        "data": args.data_config or config_root / "data.yaml",
        "models": args.model_config or config_root / "models.yaml",
        "features": args.feature_config or config_root / "features.yaml",
        "analysis": args.analysis_config or config_root / "analysis.yaml",
        "revision_roadmap": args.revision_roadmap,
        "claim_surface_manifest": args.claim_surface_manifest,
    }
    report = audit_stage4_inputs(
        manifest_path=args.manifest,
        features_dir=args.features,
        model_root=args.model_root,
        output_path=args.output,
        config_paths=config_paths,
        timestamp_tolerance_seconds=args.timestamp_tolerance_seconds,
        expected_recording_count=args.expected_recording_count,
    )
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "error_count": len(report["errors"]),
                "output": str(args.output),
            }
        )
    )
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
