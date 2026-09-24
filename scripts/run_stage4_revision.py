#!/usr/bin/env python3
"""Command-line entry point for the immutable-input Stage 4 revision."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Any

from speech_strf.stage4_runner import Stage4Runner, VARIANTS


def _fit_layer(config: str, model: str, variant: str, layer: str) -> dict[str, Any]:
    return Stage4Runner(config).fit(model, variant, layer)[0]


def _fast_refit_layer(
    config: str, model: str, layer: str, output_namespace: str
) -> dict[str, Any]:
    return Stage4Runner(config).fast_refit(
        model, layer, output_namespace=output_namespace
    )


def _fixed_null_layer(
    config: str, model: str, null_index: int, layer: str
) -> dict[str, Any]:
    return Stage4Runner(config).fixed_alpha_null(model, null_index, layer)


def _parallel_layers(function: Any, arguments: list[tuple[Any, ...]], workers: int) -> list:
    if workers not in (1, 2):
        raise ValueError("layer_workers must be one or two")
    if workers == 1 or len(arguments) == 1:
        return [function(*values) for values in arguments]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, *values) for values in arguments]
        return [future.result() for future in futures]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage4_revision.yaml", help="Stage 4 YAML config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Fit observed encoding models")
    fit.add_argument("model")
    fit.add_argument("variant", choices=VARIANTS)
    fit_layers = fit.add_mutually_exclusive_group()
    fit_layers.add_argument("--layer", help="Fit one HDF5 layer; default discovers all")
    fit_layers.add_argument("--layers", nargs="+", help="Fit only these HDF5 layers")
    fit.add_argument("--layer-workers", type=int, choices=(1, 2), default=1)

    null = subparsers.add_parser("null", help="Fit one circular-shift null index")
    null.add_argument("model")
    null.add_argument("null_index", type=int)
    count = null.add_mutually_exclusive_group(required=True)
    count.add_argument(
        "--dry-null-count",
        type=int,
        metavar="N",
        help="Nonfinal dry ensemble size (at least 20)",
    )
    count.add_argument(
        "--full-null-count",
        type=int,
        metavar="N",
        help="Final ensemble size (must be exactly 100)",
    )
    null.add_argument("--layer", help="Fit one HDF5 layer; default discovers all")

    fast = subparsers.add_parser(
        "fast-refit",
        help="Run grouped primary refits with compatible frozen original alphas",
    )
    fast.add_argument("model")
    fast.add_argument("--layers", nargs="+", required=True)
    fast.add_argument("--layer-workers", type=int, choices=(1, 2), default=1)
    fast.add_argument(
        "--output-namespace",
        default="primary_fast_refits",
        help="Flat Stage 4 output namespace; benchmark jobs use an isolated value",
    )

    fixed_null = subparsers.add_parser(
        "fixed-alpha-null",
        help="Run fixed-hyperparameter grouped null sensitivity units",
    )
    fixed_null.add_argument("model")
    fixed_null.add_argument("null_index", type=int)
    fixed_null.add_argument("--layers", nargs="+", required=True)
    fixed_null.add_argument("--layer-workers", type=int, choices=(1, 2), default=1)

    subparsers.add_parser("summarize", help="Summarize valid completed units")
    subparsers.add_parser("audit", help="Run the full immutable-input audit")
    subparsers.add_parser("synthetic-test", help="Run deterministic synthetic E2E")
    smoke = subparsers.add_parser(
        "functional-smoke",
        help="Run the gated minimal-alpha synthetic and one-recording real smoke",
    )
    smoke.add_argument("--model", default="hubert_base")
    smoke.add_argument("--layer", help="One HDF5 layer; default uses the first")
    smoke.add_argument(
        "--recording-id",
        help="One recording; default selects the first supporting a valid null shift",
    )
    return parser


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    runner = Stage4Runner(args.config)
    if args.command == "fit":
        if args.layers:
            result = _parallel_layers(
                _fit_layer,
                [
                    (args.config, args.model, args.variant, layer)
                    for layer in args.layers
                ],
                args.layer_workers,
            )
        else:
            result = runner.fit(args.model, args.variant, args.layer)
    elif args.command == "null":
        dry = args.dry_null_count is not None
        count = args.dry_null_count if dry else args.full_null_count
        result = runner.null(
            args.model,
            args.null_index,
            dry_run=dry,
            null_count=count,
            layer=args.layer,
        )
    elif args.command == "summarize":
        result = runner.summarize()
    elif args.command == "audit":
        result = runner.audit()
    elif args.command == "functional-smoke":
        result = runner.functional_smoke(
            args.model,
            layer=args.layer,
            recording_id=args.recording_id,
        )
    elif args.command == "fast-refit":
        result = _parallel_layers(
            _fast_refit_layer,
            [
                (args.config, args.model, layer, args.output_namespace)
                for layer in args.layers
            ],
            args.layer_workers,
        )
    elif args.command == "fixed-alpha-null":
        result = _parallel_layers(
            _fixed_null_layer,
            [
                (args.config, args.model, args.null_index, layer)
                for layer in args.layers
            ],
            args.layer_workers,
        )
    else:
        result = runner.synthetic_test()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
