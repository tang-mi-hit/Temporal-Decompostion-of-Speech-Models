#!/usr/bin/env python3
"""Build or verify the compact Stage 4 manuscript handoff package."""

from __future__ import annotations

import argparse
import json

from speech_strf.stage4_handoff import (
    build_stage4_handoff,
    verify_stage4_handoff,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_revision.yaml")
    parser.add_argument(
        "--output",
        default="outputs/stage4_revision/handoff",
    )
    parser.add_argument(
        "--job-ids",
        default="",
        help="Comma-separated Slurm job IDs; log filenames are also scanned",
    )
    parser.add_argument(
        "--patch-base",
        help="Optional Git revision used to generate verification/source/repository.patch",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Timestamp-archive an existing handoff before rebuilding",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing --output instead of building",
    )
    args = parser.parse_args()
    if args.verify:
        result = verify_stage4_handoff(args.output)
    else:
        result = build_stage4_handoff(
            args.config,
            output=args.output,
            job_ids=[value for value in args.job_ids.split(",") if value],
            patch_base=args.patch_base,
            archive_existing=args.archive_existing,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
