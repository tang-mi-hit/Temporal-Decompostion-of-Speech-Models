from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .design_matrix import reduced_columns
from .evaluate import conditional_contribution, held_out_r2, stable_proportion


def grouped_splits(groups: np.ndarray, n_splits: int):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two split groups are required")
    n_splits = min(n_splits, len(unique))
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def _project_targets(y_train, y_valid, analysis, seed):
    if analysis["target_mode"] != "pca":
        return y_train, y_valid
    n_components = min(
        int(analysis["n_components"]), y_train.shape[1], max(1, len(y_train) - 1)
    )
    projector = PCA(n_components=n_components, random_state=seed)
    return projector.fit_transform(y_train), projector.transform(y_valid)


def _choose_alpha(x, y, groups, analysis, seed) -> float:
    alphas = analysis["alphas"]
    scores = {float(alpha): [] for alpha in alphas}
    for train, valid in grouped_splits(groups, int(analysis["inner_folds"])):
        y_train, y_valid = _project_targets(y[train], y[valid], analysis, seed)
        for alpha in scores:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[train], y_train)
            scores[alpha].append(held_out_r2(y_valid, model.predict(x[valid])))
    return max(scores, key=lambda alpha: np.mean(scores[alpha]))


def nested_group_encoding(
    design: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    families: list[str],
    config: dict,
    layer: str = "layer_00_input",
) -> tuple[pd.DataFrame, dict]:
    analysis = config["analysis"]
    rows, kernels = [], {}
    for fold, (train, test) in enumerate(
        grouped_splits(groups, int(analysis["outer_folds"]))
    ):
        full_alpha = _choose_alpha(
            design[train],
            targets[train],
            groups[train],
            analysis,
            config["random_seed"],
        )
        y_train, y_test = _project_targets(
            targets[train], targets[test], analysis, config["random_seed"]
        )
        full = make_pipeline(StandardScaler(), Ridge(alpha=full_alpha))
        full.fit(design[train], y_train)
        full_r2 = held_out_r2(y_test, full.predict(design[test]))
        rows.append(
            {
                "layer": layer,
                "outer_fold": fold,
                "feature_family": "full",
                "alpha": full_alpha,
                "full_r2": full_r2,
                "reduced_r2": np.nan,
                "conditional_delta_r2": np.nan,
                "conditional_proportion": np.nan,
            }
        )
        kernels[f"fold_{fold}_full"] = full[-1].coef_
        for family in sorted(set(families)):
            keep = reduced_columns(families, family)
            alpha = _choose_alpha(
                design[train][:, keep],
                targets[train],
                groups[train],
                analysis,
                config["random_seed"],
            )
            reduced = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            reduced.fit(design[train][:, keep], y_train)
            reduced_r2 = held_out_r2(y_test, reduced.predict(design[test][:, keep]))
            delta = conditional_contribution(full_r2, reduced_r2)
            rows.append(
                {
                    "layer": layer,
                    "outer_fold": fold,
                    "feature_family": family,
                    "alpha": alpha,
                    "full_r2": full_r2,
                    "reduced_r2": reduced_r2,
                    "conditional_delta_r2": delta,
                    "conditional_proportion": stable_proportion(
                        delta,
                        full_r2,
                        float(analysis["stable_positive_r2_epsilon"]),
                    ),
                }
            )
    return pd.DataFrame(rows), kernels

