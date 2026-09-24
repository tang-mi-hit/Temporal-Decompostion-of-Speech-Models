#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import speech_strf.extract_features as base_feature_module
import speech_strf.stage4_features as stage4_features
import speech_strf.timebase as timebase_module
from speech_strf.alignments import Interval, parse_textgrid
from speech_strf.audio import load_standardized
from speech_strf.provenance import load_config, sha256_file, write_run_manifest
from speech_strf.stage4_features import (
    extract_stage4_features,
    verify_feature_archive,
    write_feature_archive_atomic,
)


PHONE_TIERS = ("phone", "phones", "phoneme", "phonemes")


def global_phone_categories(parsed: dict[str, list[Interval]]) -> list[str]:
    return sorted(
        {
            interval.label
            for intervals in parsed.values()
            for interval in intervals
            if interval.tier.lower() in PHONE_TIERS and interval.label.strip()
        }
    )


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recording_id(value: object) -> str:
    recording_id = str(value)
    if (
        not recording_id
        or recording_id in {".", ".."}
        or Path(recording_id).name != recording_id
    ):
        raise ValueError(f"Unsafe recording_id: {recording_id!r}")
    return recording_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/features_stage4_rich.yaml")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--validation-report", default="outputs/validation_report.json")
    parser.add_argument(
        "--output",
        help="Override config output_dir (default: outputs/stage4_revision/features_rich)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute even archives with valid integrity and matching provenance",
    )
    args = parser.parse_args()

    validation_path = Path(args.validation_report)
    if not json.loads(validation_path.read_text(encoding="utf-8"))["valid"]:
        raise SystemExit("Validation report is invalid; feature extraction refused")

    config_path = Path(args.config)
    manifest_path = Path(args.manifest)
    config = load_config(config_path)
    output = Path(args.output or config.get(
        "output_dir", "outputs/stage4_revision/features_rich"
    ))
    output.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(manifest_path).to_dict("records")
    parsed: dict[str, list[Interval]] = {}
    normalized_rows: list[tuple[str, dict]] = []
    for row in rows:
        recording_id = _recording_id(row["recording_id"])
        normalized_rows.append((recording_id, row))
        parsed[recording_id] = parse_textgrid(row["alignment_path"])

    categories = global_phone_categories(parsed)
    config["phone_categories"] = categories
    shared_hashes = {
        "config_sha256": sha256_file(config_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "validation_report_sha256": sha256_file(validation_path),
        "global_phone_categories_sha256": _json_sha256(categories),
        "extractor_source_sha256": sha256_file(stage4_features.__file__),
        "base_extractor_source_sha256": sha256_file(base_feature_module.__file__),
        "timebase_source_sha256": sha256_file(timebase_module.__file__),
        "driver_source_sha256": sha256_file(__file__),
    }

    status_counts = {"extracted": 0, "resumed": 0, "replaced_invalid": 0}
    for recording_id, row in normalized_rows:
        archive_path = output / f"{recording_id}.npz"
        record_hashes = {
            **shared_hashes,
            "audio_sha256": sha256_file(row["audio_path"]),
            "alignment_sha256": sha256_file(row["alignment_path"]),
        }
        valid, reason = verify_feature_archive(
            archive_path, expected_provenance_hashes=record_hashes
        )
        if valid and not args.no_resume:
            status_counts["resumed"] += 1
            print(f"{recording_id}: resumed (integrity verified)")
            continue

        existed = archive_path.exists() or archive_path.with_name(
            f"{archive_path.name}.sha256"
        ).exists()
        sample_rate = int(config["audio_sample_rate_hz"])
        audio, metadata = load_standardized(row["audio_path"], sample_rate)
        result = extract_stage4_features(
            audio,
            sample_rate,
            metadata["original_duration_seconds"],
            parsed[recording_id],
            config,
            provenance_hashes=record_hashes,
        )
        write_feature_archive_atomic(archive_path, result)
        verified, verification_reason = verify_feature_archive(
            archive_path, expected_provenance_hashes=record_hashes
        )
        if not verified:
            raise RuntimeError(
                f"Published archive failed verification for {recording_id}: "
                f"{verification_reason}"
            )
        if existed and not args.no_resume:
            status_counts["replaced_invalid"] += 1
            print(f"{recording_id}: replaced invalid archive ({reason})")
        else:
            status_counts["extracted"] += 1
            print(f"{recording_id}: extracted")

    write_run_manifest(
        config_path,
        output,
        input_manifest_path=manifest_path,
        extra={
            "stage": "stage4_rich",
            "output_dir": str(output),
            "global_phone_categories": categories,
            "global_phone_categories_sha256": shared_hashes[
                "global_phone_categories_sha256"
            ],
            "extractor_source_sha256": shared_hashes["extractor_source_sha256"],
            "base_extractor_source_sha256": shared_hashes[
                "base_extractor_source_sha256"
            ],
            "timebase_source_sha256": shared_hashes["timebase_source_sha256"],
            "driver_source_sha256": shared_hashes["driver_source_sha256"],
            "archive_integrity": "sha256_sidecar_verified_before_resume",
            "status_counts": status_counts,
        },
    )


if __name__ == "__main__":
    main()
