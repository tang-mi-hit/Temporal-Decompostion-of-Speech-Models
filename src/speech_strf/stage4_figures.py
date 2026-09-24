"""Publication figures built only from Stage 4 figure-source CSV tables."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelSpec:
    letter: str
    filename: str
    title: str
    effect_label: str = "ΔR²"


PANEL_SPECS: Final[tuple[PanelSpec, ...]] = (
    PanelSpec("A", "primary_original.csv", "Original feature set"),
    PanelSpec("B", "primary_rich.csv", "Rich feature set"),
    PanelSpec("C", "primary_capacity.csv", "Capacity-matched predictors"),
    PanelSpec("D", "primary_zero_lag.csv", "Zero-lag model"),
    PanelSpec(
        "E",
        "null_observed_minus_mean.csv",
        "Observed − circular-shift null",
    ),
    PanelSpec(
        "F",
        "zero_vs_five_lag.csv",
        "Zero-lag − five-lag",
        "ΔR² difference",
    ),
)

FAMILY_ORDER: Final[tuple[str, ...]] = (
    "acoustic",
    "prosodic",
    "phonetic",
    "word",
    "onset",
)

MODEL_LABELS: Final[dict[str, str]] = {
    "hubert_large_refactor_rerun": "HuBERT Large",
    "hubert_base": "HuBERT Base",
    "wav2vec2_base": "Wav2Vec 2 Base",
    "wav2vec2_large": "Wav2Vec 2 Large",
    "wavlm_base_plus": "WavLM Base+",
    "wavlm_large": "WavLM Large",
    "data2vec_audio_base": "data2vec Base",
    "xls_r_300m": "XLS-R 300M",
    "whisper_medium_encoder": "Whisper Medium",
}

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "family",
    "effect",
    "ci_low",
    "ci_high",
)


def _resolve_source_dir(source_dir: str | Path) -> Path:
    """Accept either the figure-source directory or the Stage 4 output root."""
    source = Path(source_dir)
    nested = source / "figure_sources"
    if nested.is_dir():
        return nested
    return source


def _validate_summary_status(source: Path) -> None:
    if source.name != "source_tables" or source.parent.name != "figures":
        raise ValueError(
            "Production Stage 4 figures require the configured "
            "outputs/stage4_revision/figures/source_tables directory"
        )
    status_path = source.parent.parent / "model_comparison" / "summary_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Missing or invalid complete Stage 4 summary status: {status_path}: {exc}"
        ) from exc
    if status.get("state") != "complete":
        raise ValueError(f"Stage 4 summary status is not complete: {status_path}")
    expected = {
        Path(value["path"]).resolve(): value["sha256"]
        for value in status.get("artifacts", {}).values()
    }
    expected_by_name = {path.name: digest for path, digest in expected.items()}
    for spec in PANEL_SPECS:
        path = (source / spec.filename).resolve()
        expected_digest = expected.get(path, expected_by_name.get(path.name))
        if expected_digest is None:
            raise ValueError(f"Figure source is absent from summary status: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(f"Figure source hash differs from summary status: {path}")


def _load_panel(path: Path, spec: PanelSpec) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Stage 4 figure table for panel {spec.letter} is empty: {path}"
        ) from exc
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Stage 4 figure table for panel {spec.letter} ({path.name}) "
            f"is missing required columns: {missing}"
        )
    if frame.empty:
        raise ValueError(
            f"Stage 4 figure table for panel {spec.letter} has no rows: {path}"
        )
    duplicate = frame.duplicated(["model", "family"], keep=False)
    if duplicate.any():
        cells = sorted(
            {
                (str(row.model), str(row.family))
                for row in frame.loc[duplicate, ["model", "family"]].itertuples(
                    index=False
                )
            }
        )
        raise ValueError(
            f"Stage 4 figure table for panel {spec.letter} has duplicate "
            f"model-family cells: {cells}"
        )
    unexpected = sorted(set(frame["family"].astype(str)) - set(FAMILY_ORDER))
    if unexpected:
        raise ValueError(
            f"Stage 4 figure table for panel {spec.letter} has unexpected "
            f"families: {unexpected}"
        )
    for column in ("effect", "ci_low", "ci_high"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        bad = np.flatnonzero(~np.isfinite(numeric.to_numpy(dtype=float))).tolist()
        if bad:
            raise ValueError(
                f"Stage 4 figure table for panel {spec.letter} has non-finite "
                f"{column!r} values at zero-based rows {bad}"
            )
        frame[column] = numeric
    invalid_ci = frame["ci_low"] > frame["ci_high"]
    if invalid_ci.any():
        rows = np.flatnonzero(invalid_ci.to_numpy()).tolist()
        raise ValueError(
            f"Stage 4 figure table for panel {spec.letter} has ci_low > "
            f"ci_high at zero-based rows {rows}"
        )
    return frame


def _model_order(frames: list[pd.DataFrame]) -> list[str]:
    observed = {str(value) for frame in frames for value in frame["model"]}
    preferred = [name for name in MODEL_LABELS if name in observed]
    return preferred + sorted(observed - set(preferred))


def _matrix(
    frame: pd.DataFrame,
    models: list[str],
    column: str,
) -> np.ndarray:
    indexed = frame.assign(
        model=frame["model"].astype(str),
        family=frame["family"].astype(str),
    ).set_index(["model", "family"])[column]
    return np.asarray(
        [
            [
                indexed.get((model, family), np.nan)
                for family in FAMILY_ORDER
            ]
            for model in models
        ],
        dtype=float,
    )


def _annotation(effect: float, low: float, high: float) -> str:
    if not np.isfinite(effect):
        return "—"
    return f"{effect:.3f}\n[{low:.3f}, {high:.3f}]"


def make_stage4_figures(
    source_dir: str | Path = "outputs/stage4_revision/figures/source_tables",
    output_dir: str | Path = "outputs/stage4_revision/figures",
    *,
    require_complete_status: bool = True,
) -> dict[str, Path]:
    """Create revised Figure 2 and exact per-panel source CSVs.

    The function fails closed unless all six precomputed Stage 4 tables are
    present and valid. It never reads fits, activations, manifests, or raw data.
    """
    source = _resolve_source_dir(source_dir)
    output = Path(output_dir)
    if require_complete_status:
        _validate_summary_status(source)
    paths = {spec: source / spec.filename for spec in PANEL_SPECS}
    missing = [
        f"panel {spec.letter}: {path}"
        for spec, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot create revised Figure 2; missing required Stage 4 "
            "figure-source tables:\n- " + "\n- ".join(missing)
        )

    frames = [_load_panel(paths[spec], spec) for spec in PANEL_SPECS]
    models = _model_order(frames)
    effects = [_matrix(frame, models, "effect") for frame in frames]
    finite = np.concatenate([values[np.isfinite(values)] for values in effects])
    if finite.size == 0:
        raise ValueError("Stage 4 figure tables contain no finite effects")
    color_min = float(finite.min())
    color_max = float(finite.max())
    if color_min == color_max:
        padding = max(abs(color_min) * 0.05, 1e-9)
        color_min -= padding
        color_max += padding
    normalization = Normalize(vmin=color_min, vmax=color_max)

    # ICASSP two-column width with a near-page-height layout. The 2×3 grid
    # leaves enough room for model names and two-line numerical annotations.
    with plt.rc_context(
        {
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(7.16, 8.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        images = []
        for axis, spec, frame, effect in zip(
            axes.flat, PANEL_SPECS, frames, effects
        ):
            low = _matrix(frame, models, "ci_low")
            high = _matrix(frame, models, "ci_high")
            image = axis.imshow(
                np.ma.masked_invalid(effect),
                cmap="Blues",
                norm=normalization,
                aspect="auto",
                interpolation="nearest",
            )
            images.append(image)
            axis.set_title(f"{spec.letter}. {spec.title}", loc="left", pad=4)
            axis.set_xticks(range(len(FAMILY_ORDER)))
            axis.set_xticklabels(
                [family.capitalize() for family in FAMILY_ORDER],
                rotation=32,
                ha="right",
                rotation_mode="anchor",
            )
            axis.set_yticks(range(len(models)))
            axis.set_yticklabels(
                [MODEL_LABELS.get(model, model.replace("_", " ")) for model in models]
            )
            axis.tick_params(length=0)
            axis.set_xticks(
                np.arange(-0.5, len(FAMILY_ORDER), 1), minor=True
            )
            axis.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.55)
            axis.tick_params(which="minor", bottom=False, left=False)
            for row in range(len(models)):
                for column in range(len(FAMILY_ORDER)):
                    value = effect[row, column]
                    text_color = (
                        "white"
                        if np.isfinite(value)
                        and normalization(value) > 0.62
                        else "#202020"
                    )
                    axis.text(
                        column,
                        row,
                        _annotation(value, low[row, column], high[row, column]),
                        ha="center",
                        va="center",
                        fontsize=5.3,
                        color=text_color,
                        linespacing=0.94,
                    )

        axes[0, 0].legend(
            handles=[
                Patch(facecolor=plt.cm.Blues(0.72), label="Mean effect (cell shade)"),
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=r"$[\,]$",
                    color="#202020",
                    label="95% recording-bootstrap CI",
                ),
            ],
            loc="upper left",
            bbox_to_anchor=(0, 1.02),
            frameon=True,
            borderpad=0.35,
            handlelength=1.2,
        )
        colorbar = figure.colorbar(
            images[0],
            ax=axes,
            location="right",
            fraction=0.026,
            pad=0.015,
        )
        colorbar.set_label("Effect size (darker = larger)")
        figure.suptitle(
            "Candidate revised Figure 2 — conditional unique contribution",
            fontsize=9,
        )

        output.mkdir(parents=True, exist_ok=True)
        pdf = output / "candidate_revised_figure_2.pdf"
        png = output / "candidate_revised_figure_2.png"
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(figure)

    result: dict[str, Path] = {"pdf": pdf, "png": png}
    for spec in PANEL_SPECS:
        destination = output / f"candidate_revised_figure_2_panel_{spec.letter}.csv"
        shutil.copyfile(paths[spec], destination)
        result[f"panel_{spec.letter}"] = destination
    return result


__all__ = ["PANEL_SPECS", "make_stage4_figures"]
