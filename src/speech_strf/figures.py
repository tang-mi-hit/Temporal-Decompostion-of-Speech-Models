from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alignments import Interval
from .statistics import layer_summary


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def make_result_figures(
    results: pd.DataFrame,
    kernels: dict,
    output_dir: str | Path,
    title_prefix: str = "",
) -> None:
    prefix = f"{title_prefix}: " if title_prefix else ""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = layer_summary(results)
    summary.to_csv(output / "layer_feature_summary.csv", index=False)
    results.to_csv(output / "outer_fold_results.csv", index=False)
    contributions = summary[summary.feature_family != "full"]
    heat = contributions.pivot(
        index="layer", columns="feature_family", values="mean_conditional_delta_r2"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(heat.fillna(0), aspect="auto", cmap="coolwarm")
    ax.set(xticks=np.arange(len(heat.columns)), xticklabels=heat.columns, ylabel="layer")
    ax.set_yticks(np.arange(len(heat.index)), labels=heat.index)
    ax.set_title(f"{prefix}Held-out conditional unique contribution (ΔR²)")
    fig.colorbar(image, ax=ax)
    _save(fig, output / "layer_feature_heatmap")

    full = summary[summary.feature_family == "full"]
    fig, ax = plt.subplots()
    ax.plot(full.layer, full.mean_full_r2, marker="o")
    ax.set(
        xlabel="Layer",
        ylabel="Held-out R²",
        title=f"{prefix}Full-model layerwise performance",
    )
    _save(fig, output / "layerwise_full_r2")

    fig, ax = plt.subplots()
    for family, rows in contributions.groupby("feature_family"):
        ax.plot(rows.layer, rows.mean_conditional_delta_r2, marker="o", label=family)
    ax.set(
        xlabel="Layer",
        ylabel="Held-out ΔR²",
        title=f"{prefix}Conditional contribution by family",
    )
    ax.legend()
    _save(fig, output / "layerwise_contributions")

    fig, ax = plt.subplots()
    fold_full = results[results.feature_family == "full"]
    ax.scatter(fold_full.outer_fold, fold_full.full_r2)
    ax.axhline(fold_full.full_r2.mean(), color="black", linestyle="--")
    ax.set(
        xlabel="Outer fold",
        ylabel="Held-out R²",
        title=f"{prefix}Held-out-fold reliability",
    )
    _save(fig, output / "fold_reliability")

    key = sorted(kernels)[0]
    kernel = np.asarray(kernels[key])
    fig, ax = plt.subplots()
    ax.plot(kernel[0] if kernel.ndim == 2 else kernel)
    ax.set(
        xlabel="Lagged predictor index",
        ylabel="Ridge coefficient",
        title=f"{prefix}Diagnostic kernel: {key}",
    )
    _save(fig, output / "diagnostic_kernel")


def alignment_diagnostic(
    times: np.ndarray,
    waveform: np.ndarray,
    sample_rate: int,
    envelope: np.ndarray,
    intervals: list[Interval],
    activation_times: np.ndarray,
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    wave_times = np.arange(len(waveform)) / sample_rate
    ax.plot(wave_times, waveform / (np.max(np.abs(waveform)) + 1e-12), alpha=.35)
    ax.plot(times, envelope / (np.max(envelope) + 1e-12), label="envelope")
    for row in intervals:
        color = "tab:red" if row.tier.lower().startswith("word") else "tab:green"
        ax.axvline(row.start, color=color, alpha=.25)
    ax.scatter(activation_times, np.full_like(activation_times, -1), s=2, label="activation frames")
    ax.set(xlabel="Audio-relative seconds", title="Timebase alignment diagnostic")
    ax.legend()
    _save(fig, Path(output_path).with_suffix(""))

