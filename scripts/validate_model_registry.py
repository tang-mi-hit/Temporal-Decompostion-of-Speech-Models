#!/usr/bin/env python3
from __future__ import annotations

import argparse

from speech_strf.model_registry import load_model_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument(
        "--check-local",
        action="store_true",
        help="Also require every enabled checkpoint to exist in the local HF cache/path",
    )
    args = parser.parse_args()
    defaults, entries = load_model_registry(args.config)
    import transformers

    failures = []
    print(f"Registry valid: {len(entries)} models")
    print(f"Default model: {defaults['default_model']}")
    for key, entry in entries.items():
        state = "enabled" if entry.enabled else "disabled"
        if not hasattr(transformers, entry.loading_class):
            failures.append(
                f"{key}: Transformers {transformers.__version__} has no "
                f"{entry.loading_class}"
            )
        if args.check_local and entry.enabled:
            try:
                transformers.AutoConfig.from_pretrained(
                    entry.model_id,
                    revision=entry.revision,
                    local_files_only=True,
                )
            except Exception as exc:
                failures.append(f"{key}: local checkpoint validation failed: {exc}")
        print(
            f"{key}: {entry.model_id} [{entry.input_modality}, "
            f"{entry.adapter}, {state}]"
        )
    if failures:
        raise SystemExit("Registry validation failures:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()

