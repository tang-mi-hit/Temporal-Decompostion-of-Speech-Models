#!/usr/bin/env python
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from speech_strf.figures import make_result_figures
from speech_strf.provenance import write_run_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", default="outputs/fit/results.csv")
    parser.add_argument("--kernels", default="outputs/fit/kernels.npz")
    parser.add_argument("--output", default="outputs/figures")
    args = parser.parse_args()
    kernels_file = np.load(args.kernels)
    make_result_figures(
        pd.read_csv(args.results),
        {key: kernels_file[key] for key in kernels_file.files},
        args.output,
    )
    write_run_manifest(args.config, args.output)


if __name__ == "__main__":
    main()

