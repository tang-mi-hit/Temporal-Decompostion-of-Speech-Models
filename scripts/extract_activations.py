#!/usr/bin/env python3
"""Locked legacy entry point retained to prevent accidental reference overwrites."""

raise SystemExit(
    "The legacy HuBERT extractor is locked because outputs/activations.h5 is a "
    "completed reference artifact. Use scripts/run_model_pipeline.py; it writes "
    "to outputs/<model-key>/."
)
