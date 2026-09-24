#!/usr/bin/env python3
"""Create candidate revised Figure 2 from Stage 4 figure-source CSVs."""

from __future__ import annotations

import argparse
import json

from speech_strf.stage4_figures import make_stage4_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="outputs/stage4_revision/figures/source_tables",
        help="Stage 4 figure-source directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage4_revision/figures",
        help="Figure output directory (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_parser().parse_args(argv)
    outputs = make_stage4_figures(args.source_dir, args.output_dir)
    serialized = {name: str(path) for name, path in outputs.items()}
    print(json.dumps(serialized, indent=2, sort_keys=True))
    return serialized


if __name__ == "__main__":
    main()
