#!/usr/bin/env python3
"""Validate persisted Stage 4 task layers against current activation stores."""

from __future__ import annotations

import argparse
import json

from speech_strf.stage4_compute_scope import validate_compute_scope_manifests
from speech_strf.stage4_runner import Stage4Runner, discover_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_revision.yaml")
    parser.add_argument(
        "--manifest-dir",
        default="outputs/stage4_revision/manifests/deadline_compute_scope",
    )
    args = parser.parse_args()
    runner = Stage4Runner(args.config)
    runner._audit()
    model_layers = {
        model: discover_layers(runner._model(model) / "activations.h5")
        for model in runner.model_names
    }
    result = validate_compute_scope_manifests(
        runner.resolve(args.manifest_dir),
        model_layers=model_layers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
