#!/usr/bin/env python3
"""Fit exactly the selected-depth control panels described in the paper."""

import argparse
import csv

from speech_strf.stage4_runner import Stage4Runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_analysis.yaml")
    parser.add_argument("analysis", choices=("observed", "null"))
    args = parser.parse_args()
    runner = Stage4Runner(args.config)
    manifests = runner.output_root / "manifests" / "deadline_compute_scope"
    if args.analysis == "observed":
        with (manifests / "primary_fast_refits.tsv").open(newline="") as source:
            for task in csv.DictReader(source, delimiter="\t"):
                for layer in task["layers"].split(","):
                    runner.fast_refit(task["model"], layer)
        with (manifests / "selected_depth_controls.tsv").open(newline="") as source:
            for task in csv.DictReader(source, delimiter="\t"):
                for depth in ("input", "middle", "final"):
                    runner.fit(task["model"], task["variant"], task[depth])
    else:
        with (manifests / "structured_nulls.tsv").open(newline="") as source:
            for task in csv.DictReader(source, delimiter="\t"):
                runner.fixed_alpha_null(
                    task["model"], int(task["shift"]), task["layer"]
                )


if __name__ == "__main__":
    main()
