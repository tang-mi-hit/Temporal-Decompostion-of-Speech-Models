#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from speech_strf.design_matrix import lagged_design
from speech_strf.fit_encoding import nested_group_encoding
from speech_strf.provenance import load_config, write_run_manifest
from speech_strf.timebase import resample_continuous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", default="outputs/features")
    parser.add_argument("--activations", default="outputs/activations.h5")
    parser.add_argument("--layer", default="layer_00_input")
    parser.add_argument("--output", default="outputs/fit")
    args = parser.parse_args()
    config = load_config(args.config)
    xs, ys, groups, families = [], [], [], None
    with h5py.File(args.activations) as store:
        for recording_id in sorted(store):
            feature = np.load(Path(args.features) / f"{recording_id}.npz")
            x, times = feature["matrix"], feature["times"]
            current_families = feature["families"].tolist()
            if families is not None and current_families != families:
                raise ValueError("Feature columns differ across recordings")
            families = current_families
            group = store[recording_id]
            if not group.attrs.get("complete", False):
                raise RuntimeError(f"Incomplete activation group: {recording_id}")
            activation_times = group["_frame_times_seconds"][:]
            y = resample_continuous(group[args.layer][:], activation_times, times)
            xs.append(x)
            ys.append(y)
            groups.extend([recording_id] * len(x))
    design, lagged_families = lagged_design(
        np.vstack(xs),
        np.asarray(groups),
        families,
        config["analysis"]["lags_seconds"],
        config["analysis_rate_hz"],
    )
    results, kernels = nested_group_encoding(
        design, np.vstack(ys), np.asarray(groups), lagged_families, config, args.layer
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "results.csv", index=False)
    np.savez_compressed(output / "kernels.npz", **kernels)
    write_run_manifest(args.config, output)


if __name__ == "__main__":
    main()

