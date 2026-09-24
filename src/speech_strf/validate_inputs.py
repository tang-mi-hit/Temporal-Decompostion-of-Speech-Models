from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .alignments import parse_textgrid, validate_intervals
from .audio import inspect_audio
from .manifest import manifest_hash


def validate_manifest(
    frame: pd.DataFrame, empty_end_tolerance_seconds: float = 0.0
) -> dict:
    issues: list[dict] = []
    warnings: list[dict] = []
    duplicated = frame[frame["recording_id"].duplicated(keep=False)]
    for record_id in duplicated["recording_id"]:
        issues.append(
            {"recording_id": record_id, "severity": "error", "code": "duplicate_id"}
        )
    records = []
    for row in frame.to_dict("records"):
        record_issues = []
        record_warnings = []
        if row.get("audio_matches") != 1:
            record_issues.append(
                {"severity": "error", "code": "missing_or_duplicate_audio"}
            )
        if row.get("alignment_matches") != 1:
            record_issues.append(
                {"severity": "error", "code": "missing_or_duplicate_alignment"}
            )
        metadata = None
        if not record_issues:
            try:
                metadata = inspect_audio(row["audio_path"])
                intervals = parse_textgrid(row["alignment_path"])
                findings = validate_intervals(
                    intervals,
                    metadata["duration_seconds"],
                    empty_end_tolerance_seconds,
                )
                record_issues.extend(
                    finding for finding in findings if finding["severity"] == "error"
                )
                record_warnings.extend(
                    finding for finding in findings if finding["severity"] == "warning"
                )
            except Exception as exc:
                record_issues.append(
                    {"severity": "error", "code": "read_error", "detail": str(exc)}
                )
        issues.extend({"recording_id": row["recording_id"], **x} for x in record_issues)
        warnings.extend(
            {"recording_id": row["recording_id"], **x} for x in record_warnings
        )
        records.append(
            {
                "recording_id": row["recording_id"],
                "audio_filename": row.get("audio_filename"),
                "alignment_filename": row.get("alignment_filename"),
                "audio_metadata": metadata,
                "issue_count": len(record_issues),
                "warning_count": len(record_warnings),
            }
        )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash(frame),
        "valid": not issues,
        "record_count": len(frame),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "records": records,
        "issues": issues,
        "warnings": warnings,
    }


def write_validation_report(
    frame: pd.DataFrame,
    path: str | Path,
    empty_end_tolerance_seconds: float = 0.0,
) -> dict:
    report = validate_manifest(frame, empty_end_tolerance_seconds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

