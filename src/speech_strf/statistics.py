from __future__ import annotations

import pandas as pd


def layer_summary(results: pd.DataFrame) -> pd.DataFrame:
    columns = ["layer", "feature_family"]
    return (
        results.groupby(columns, as_index=False)
        .agg(
            mean_full_r2=("full_r2", "mean"),
            mean_conditional_delta_r2=("conditional_delta_r2", "mean"),
            fold_sd_delta_r2=("conditional_delta_r2", "std"),
            n_outer_folds=("outer_fold", "nunique"),
        )
        .sort_values(columns)
    )

