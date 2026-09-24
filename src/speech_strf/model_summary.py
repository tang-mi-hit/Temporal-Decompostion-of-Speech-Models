from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_COLUMNS = {
    "layer",
    "outer_fold",
    "feature_family",
    "full_r2",
    "conditional_delta_r2",
}


def summarize_predictability(
    results_by_model: dict[str, pd.DataFrame],
    model_metadata: dict[str, dict],
    reference_key: str = "hubert_large_reference",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if reference_key not in results_by_model:
        raise ValueError(f"Missing reference results for {reference_key}")
    layer_rows = []
    family_layer_rows = []
    for model_key, results in results_by_model.items():
        missing = RESULT_COLUMNS - set(results)
        if missing:
            raise ValueError(f"{model_key} results lack columns {sorted(missing)}")
        full = results[results["feature_family"] == "full"]
        if full.empty:
            raise ValueError(f"{model_key} has no full-model rows")
        layer_order = list(dict.fromkeys(results["layer"].astype(str)))
        relative_depth = {
            layer: index / max(len(layer_order) - 1, 1)
            for index, layer in enumerate(layer_order)
        }
        for layer, rows in full.groupby("layer", sort=False):
            values = rows["full_r2"].astype(float)
            layer_rows.append(
                {
                    "model_key": model_key,
                    "model_id": model_metadata[model_key]["model_id"],
                    "modality": model_metadata[model_key]["input_modality"],
                    "layer": layer,
                    "relative_depth": relative_depth[str(layer)],
                    "mean_full_r2": values.mean(),
                    "sd_full_r2_across_folds": values.std(ddof=1),
                    "sem_full_r2_across_folds": values.sem(ddof=1),
                    "n_outer_folds": values.count(),
                }
            )
        contributions = results[results["feature_family"] != "full"]
        for (layer, family), rows in contributions.groupby(
            ["layer", "feature_family"], sort=False
        ):
            values = rows["conditional_delta_r2"].astype(float)
            family_layer_rows.append(
                {
                    "model_key": model_key,
                    "model_id": model_metadata[model_key]["model_id"],
                    "modality": model_metadata[model_key]["input_modality"],
                    "layer": layer,
                    "relative_depth": relative_depth[str(layer)],
                    "feature_family": family,
                    "mean_conditional_delta_r2": values.mean(),
                    "sd_delta_r2_across_folds": values.std(ddof=1),
                    "n_outer_folds": values.count(),
                }
            )
    layer_table = pd.DataFrame(layer_rows)
    family_layer_table = pd.DataFrame(family_layer_rows)
    family_table = (
        family_layer_table.groupby(
            ["model_key", "model_id", "modality", "feature_family"],
            as_index=False,
        )
        .agg(
            mean_delta_r2_across_layers=("mean_conditional_delta_r2", "mean"),
            peak_delta_r2=("mean_conditional_delta_r2", "max"),
            layer_sd_delta_r2=("mean_conditional_delta_r2", "std"),
            n_layers=("layer", "nunique"),
        )
    )
    peak_layers = family_layer_table.loc[
        family_layer_table.groupby(
            ["model_key", "feature_family"]
        )["mean_conditional_delta_r2"].idxmax(),
        ["model_key", "feature_family", "layer"],
    ].rename(columns={"layer": "peak_delta_r2_layer"})
    family_table = family_table.merge(
        peak_layers, on=["model_key", "feature_family"], how="left"
    )
    reference_families = family_table[
        family_table["model_key"] == reference_key
    ][["feature_family", "mean_delta_r2_across_layers", "peak_delta_r2"]].rename(
        columns={
            "mean_delta_r2_across_layers": "reference_mean_delta_r2_across_layers",
            "peak_delta_r2": "reference_peak_delta_r2",
        }
    )
    family_table = family_table.merge(reference_families, on="feature_family", how="left")
    family_table["mean_delta_r2_difference_vs_hubert"] = (
        family_table["mean_delta_r2_across_layers"]
        - family_table["reference_mean_delta_r2_across_layers"]
    )
    family_table["peak_delta_r2_difference_vs_hubert"] = (
        family_table["peak_delta_r2"] - family_table["reference_peak_delta_r2"]
    )

    model_rows = []
    best_fold_rows = []
    reference_layers = layer_table[layer_table["model_key"] == reference_key]
    reference_best = reference_layers.loc[reference_layers["mean_full_r2"].idxmax()]
    reference_mean = reference_layers["mean_full_r2"].mean()
    reference_results = results_by_model[reference_key]
    reference_best_folds = reference_results[
        (reference_results["feature_family"] == "full")
        & (reference_results["layer"] == reference_best["layer"])
    ][["outer_fold", "full_r2"]].rename(columns={"full_r2": "hubert_full_r2"})
    for model_key, rows in layer_table.groupby("model_key", sort=False):
        best = rows.loc[rows["mean_full_r2"].idxmax()]
        mean_layers = rows["mean_full_r2"].mean()
        candidate_best_folds = results_by_model[model_key][
            (results_by_model[model_key]["feature_family"] == "full")
            & (results_by_model[model_key]["layer"] == best["layer"])
        ][["outer_fold", "full_r2"]].rename(columns={"full_r2": "model_full_r2"})
        paired = candidate_best_folds.merge(reference_best_folds, on="outer_fold")
        if set(candidate_best_folds["outer_fold"]) != set(
            reference_best_folds["outer_fold"]
        ):
            raise ValueError(
                f"{model_key} and HuBERT best-layer rows do not share identical folds"
            )
        paired["full_r2_difference_vs_hubert"] = (
            paired["model_full_r2"] - paired["hubert_full_r2"]
        )
        paired.insert(0, "model_key", model_key)
        best_fold_rows.append(paired)
        model_rows.append(
            {
                "model_key": model_key,
                "model_id": best["model_id"],
                "modality": best["modality"],
                "n_layers": rows["layer"].nunique(),
                "n_outer_folds": int(best["n_outer_folds"]),
                "best_layer": best["layer"],
                "best_layer_relative_depth": best["relative_depth"],
                "best_layer_mean_full_r2": best["mean_full_r2"],
                "best_layer_sd_full_r2": best["sd_full_r2_across_folds"],
                "mean_full_r2_across_layers": mean_layers,
                "best_full_r2_difference_vs_hubert": (
                    best["mean_full_r2"] - reference_best["mean_full_r2"]
                ),
                "mean_layer_r2_difference_vs_hubert": mean_layers - reference_mean,
                "paired_fold_mean_difference_vs_hubert": paired[
                    "full_r2_difference_vs_hubert"
                ].mean(),
                "paired_fold_sd_difference_vs_hubert": paired[
                    "full_r2_difference_vs_hubert"
                ].std(ddof=1),
                "interpretation": (
                    "descriptive predictability of this model's representation; "
                    "not general model quality"
                ),
            }
        )
    model_table = pd.DataFrame(model_rows)
    best_fold_table = pd.concat(best_fold_rows, ignore_index=True)
    family_wide = family_table.pivot(
        index="model_key",
        columns="feature_family",
        values=["mean_delta_r2_across_layers", "mean_delta_r2_difference_vs_hubert"],
    )
    family_wide.columns = [
        f"{metric}__{family}" for metric, family in family_wide.columns
    ]
    model_table = model_table.merge(
        family_wide.reset_index(), on="model_key", how="left"
    )
    return model_table, layer_table, family_table, best_fold_table


def save_predictability_summary(
    output_dir: str | Path,
    model_table: pd.DataFrame,
    layer_table: pd.DataFrame,
    family_table: pd.DataFrame,
    best_fold_table: pd.DataFrame,
    source_paths: dict[str, str | Path],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_table.to_csv(output / "model_comparison_summary.csv", index=False)
    layer_table.to_csv(output / "model_layer_summary.csv", index=False)
    family_table.to_csv(output / "model_family_summary.csv", index=False)
    best_fold_table.to_csv(output / "paired_best_layer_fold_differences.csv", index=False)

    labels = model_table["model_key"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.5))
    ax.bar(
        x - 0.18,
        model_table["best_layer_mean_full_r2"],
        width=0.36,
        yerr=model_table["best_layer_sd_full_r2"],
        label="best layer",
    )
    ax.bar(
        x + 0.18,
        model_table["mean_full_r2_across_layers"],
        width=0.36,
        label="mean across layers",
    )
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Held-out R²",
        title="Cross-model representation predictability",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "model_predictability.svg")
    fig.savefig(output / "model_predictability.pdf")
    plt.close(fig)

    heat = family_table.pivot(
        index="model_key",
        columns="feature_family",
        values="mean_delta_r2_across_layers",
    )
    fig, ax = plt.subplots(figsize=(8, max(4, len(heat) * 0.5)))
    image = ax.imshow(heat, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(heat.columns)), labels=heat.columns, rotation=45)
    ax.set_yticks(np.arange(len(heat.index)), labels=heat.index)
    ax.set_title("Mean conditional unique contribution across layers")
    fig.colorbar(image, ax=ax, label="Held-out ΔR²")
    fig.tight_layout()
    fig.savefig(output / "model_family_contributions.svg")
    fig.savefig(output / "model_family_contributions.pdf")
    plt.close(fig)

    metadata = {
        "reference_model": "hubert_large_reference",
        "interpretation": (
            "Descriptive predictability of each representation by the fixed predictor "
            "set; differences do not establish general model superiority or causality."
        ),
        "source_sha256": {
            key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for key, path in source_paths.items()
        },
    }
    (output / "summary_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

