from __future__ import annotations

import struct

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

from speech_strf.stage4_figures import PANEL_SPECS, make_stage4_figures


def _write_figure_sources(root):
    source = root / "figure_sources"
    source.mkdir(parents=True)
    families = ["acoustic", "prosodic", "phonetic", "word", "onset"]
    original_bytes = {}
    for panel_index, spec in enumerate(PANEL_SPECS):
        rows = []
        for model_index, model in enumerate(("hubert_base", "wavlm_large")):
            for family_index, family in enumerate(families):
                effect = (
                    0.001
                    + panel_index * 0.002
                    + model_index * 0.0005
                    + family_index * 0.0002
                )
                rows.append(
                    {
                        "variant": spec.filename.removesuffix(".csv"),
                        "model": model,
                        "family": family,
                        "effect": effect,
                        "ci_low": effect - 0.0003,
                        "ci_high": effect + 0.0003,
                        "p_value": 0.03125,
                    }
                )
        path = source / spec.filename
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.7f")
        original_bytes[spec.letter] = path.read_bytes()
    return source, original_bytes


def _png_dimensions(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_figure_outputs_are_readable_and_panel_csvs_are_exact(
    tmp_path, monkeypatch
):
    source, original_bytes = _write_figure_sources(tmp_path)
    output = tmp_path / "figures"
    visible_titles = []
    original_set_title = matplotlib.axes.Axes.set_title

    def capture_title(axis, label, *args, **kwargs):
        visible_titles.append(label)
        return original_set_title(axis, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", capture_title)
    result = make_stage4_figures(source, output, require_complete_status=False)

    assert result["pdf"].read_bytes().startswith(b"%PDF")
    width, height = _png_dimensions(result["png"])
    assert width >= 2_000
    assert height >= 2_000
    assert [title[:2] for title in visible_titles] == [
        "A.",
        "B.",
        "C.",
        "D.",
        "E.",
        "F.",
    ]
    assert "descriptive" not in visible_titles[-1].lower()
    assert "Zero-lag − five-lag" in visible_titles[-1]
    for spec in PANEL_SPECS:
        panel_path = result[f"panel_{spec.letter}"]
        assert panel_path.parent == output
        assert panel_path.read_bytes() == original_bytes[spec.letter]


def test_effect_tables_retain_uncertainty_and_use_a_shared_sequential_scale(
    tmp_path, monkeypatch
):
    source, _ = _write_figure_sources(tmp_path)
    image_calls = []
    original_imshow = matplotlib.axes.Axes.imshow

    def capture_imshow(axis, values, *args, **kwargs):
        image_calls.append(kwargs)
        return original_imshow(axis, values, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", capture_imshow)
    result = make_stage4_figures(
        source, tmp_path / "figures", require_complete_status=False
    )

    assert len(image_calls) == 6
    assert {call["cmap"] for call in image_calls} == {"Blues"}
    assert len({id(call["norm"]) for call in image_calls}) == 1
    panel_a = pd.read_csv(result["panel_A"])
    assert {"effect", "ci_low", "ci_high"} <= set(panel_a)
    assert np.all(panel_a["ci_low"] <= panel_a["effect"])
    assert np.all(panel_a["effect"] <= panel_a["ci_high"])


def test_missing_table_fails_with_panel_and_full_path(tmp_path):
    source, _ = _write_figure_sources(tmp_path)
    missing_spec = PANEL_SPECS[4]
    missing = source / missing_spec.filename
    missing.unlink()

    with pytest.raises(FileNotFoundError) as exc:
        make_stage4_figures(
            source, tmp_path / "figures", require_complete_status=False
        )

    message = str(exc.value)
    assert f"panel {missing_spec.letter}" in message
    assert str(missing) in message
    assert not (tmp_path / "figures").exists()


def test_malformed_table_fails_before_writing_outputs(tmp_path):
    source, _ = _write_figure_sources(tmp_path)
    malformed_spec = PANEL_SPECS[1]
    malformed = source / malformed_spec.filename
    pd.read_csv(malformed).drop(columns=["ci_high"]).to_csv(
        malformed, index=False
    )

    with pytest.raises(ValueError, match=r"panel B.*ci_high"):
        make_stage4_figures(
            source, tmp_path / "figures", require_complete_status=False
        )

    assert not (tmp_path / "figures").exists()


def test_production_source_requires_complete_hashed_summary_status(tmp_path):
    source, _ = _write_figure_sources(tmp_path)
    production = tmp_path / "stage4_revision" / "figures" / "source_tables"
    production.parent.mkdir(parents=True)
    source.rename(production)

    with pytest.raises(ValueError, match="summary status"):
        make_stage4_figures(production, production.parent)
