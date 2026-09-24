#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from speech_strf.manifest import build_manifest
from speech_strf.provenance import load_config, write_run_manifest
from speech_strf.validate_inputs import write_validation_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)["data"]
    manifest = build_manifest(
        config["audio_root"], config["alignment_root"], config.get("sample_ids")
    )
    destination = Path(config["manifest_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    report = write_validation_report(
        manifest,
        config["validation_report"],
        float(config.get("empty_annotation_end_tolerance_seconds", 0.0)),
    )
    write_run_manifest(args.config, destination.parent, input_manifest_path=destination)
    print(
        f"Wrote {destination} and {config['validation_report']}; "
        f"valid={report['valid']}; errors={report['issue_count']}; "
        f"warnings={report['warning_count']}"
    )
    raise SystemExit(0 if report["valid"] else 2)


if __name__ == "__main__":
    main()

