#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from speech_strf.alignments import parse_textgrid
from speech_strf.audio import load_standardized
from speech_strf.extract_features import extract_features
from speech_strf.provenance import load_config, write_run_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--validation-report", default="outputs/validation_report.json")
    parser.add_argument("--output", default="outputs/features")
    args = parser.parse_args()
    if not json.loads(Path(args.validation_report).read_text())["valid"]:
        raise SystemExit("Validation report is invalid; feature extraction refused")
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.manifest).to_dict("records")
    parsed = {row["recording_id"]: parse_textgrid(row["alignment_path"]) for row in rows}
    config["phone_categories"] = sorted(
        {
            interval.label
            for intervals in parsed.values()
            for interval in intervals
            if interval.tier.lower() in ("phone", "phones", "phoneme", "phonemes")
            and interval.label.strip()
        }
    )
    for row in rows:
        sample_rate = int(config["audio_sample_rate_hz"])
        audio, metadata = load_standardized(row["audio_path"], sample_rate)
        result = extract_features(
            audio,
            sample_rate,
            metadata["original_duration_seconds"],
            parsed[row["recording_id"]],
            config,
        )
        np.savez_compressed(
            output / f"{row['recording_id']}.npz",
            matrix=result["matrix"],
            times=result["times"],
            names=np.asarray(result["names"]),
            families=np.asarray(result["families"]),
            log=json.dumps(result["log"]),
        )
    write_run_manifest(args.config, output, input_manifest_path=args.manifest)


if __name__ == "__main__":
    main()

