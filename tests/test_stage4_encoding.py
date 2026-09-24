from __future__ import annotations

import json

import h5py
import numpy as np

from speech_strf.design_matrix import lagged_design
from speech_strf.stage4_encoding import (
    FAMILIES,
    _score,
    fit_stage4_encoding,
    fit_stage4_fixed_alpha_grouped,
    load_stage4_recordings,
)
from speech_strf.fit_encoding import grouped_splits


def _recordings(count=4, frames=12, target_width=4):
    rng = np.random.default_rng(42)
    recordings = {}
    families = ["acoustic", "acoustic", "prosodic", "phonetic", "word", "onset"]
    coefficient = rng.normal(size=(len(families), target_width))
    for index in range(count):
        matrix = rng.normal(size=(frames, len(families)))
        targets = matrix @ coefficient + 0.05 * rng.normal(size=(frames, target_width))
        recordings[f"r{index}"] = {
            "matrix": matrix,
            "targets": targets,
            "times": np.arange(frames) / 50,
            "families": families,
            "duration_seconds": frames / 50,
        }
    return recordings


def test_canonical_npz_hdf5_loader_and_recording_level_outputs(tmp_path):
    source = _recordings()
    features = tmp_path / "features"
    features.mkdir()
    store_path = tmp_path / "activations.h5"
    with h5py.File(store_path, "w") as store:
        for recording_id, record in source.items():
            np.savez_compressed(
                features / f"{recording_id}.npz",
                matrix=record["matrix"].astype(np.float32),
                times=record["times"],
                names=np.array([f"feature_{i}" for i in range(6)]),
                families=np.array(record["families"]),
                log=json.dumps({}),
            )
            group = store.create_group(recording_id)
            group.attrs["complete"] = True
            group.create_dataset("canonical_timestamps", data=record["times"])
            group.create_group("canonical").create_dataset(
                "layer_00_input", data=record["targets"].astype(np.float32)
            )

    loaded = load_stage4_recordings(features, store_path, "layer_00_input")
    result = fit_stage4_encoding(loaded, alphas=[0.1, 1.0])

    scores = result["scores"]
    primary = scores.query("split_kind == 'primary'")
    sensitivity = scores.query("split_kind == 'sensitivity'")
    assert len(primary) == len(source) * len(FAMILIES)
    assert len(sensitivity) == len(source) * len(FAMILIES)
    assert set(scores["recording_id"]) == set(source)
    assert set(scores["family"]) == set(FAMILIES)
    assert (primary.groupby("recording_id").size() == len(FAMILIES)).all()
    assert set(scores["frames"]) == {12}
    assert set(scores["duration_seconds"]) == {12 / 50}
    assert np.allclose(
        scores["delta_r2"], scores["full_r2"] - scores["reduced_r2"]
    )
    assert set(result["predictions"]) == set(source)
    for recording_id in source:
        assert set(result["predictions"][recording_id]) == set(FAMILIES)
        for family in FAMILIES:
            prediction = result["predictions"][recording_id][family]
            assert prediction["full"].shape == (12, 4)
            assert prediction["reduced"].shape == (12, 4)


def test_outer_and_inner_fits_never_include_held_recording(monkeypatch):
    records = _recordings(frames=10)
    records["r3"]["matrix"][:] = 1e12
    records["r3"]["targets"][:] = 1e12
    observed_fit_maxima = []

    from speech_strf import stage4_encoding

    original_fit = stage4_encoding.StandardScaler.fit

    def recording_fit(self, values, *args, **kwargs):
        observed_fit_maxima.append(float(np.max(np.abs(values))))
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(stage4_encoding.StandardScaler, "fit", recording_fit)
    result = fit_stage4_encoding(records, alphas=[1.0])

    r3_fold = next(
        report
        for report in result["split_reports"]
        if report["split_kind"] == "primary"
        and report["test_recording_ids"] == ["r3"]
    )
    assert "r3" not in r3_fold["train_recording_ids"]
    for report in result["split_reports"]:
        assert not (
            set(report["train_recording_ids"]) & set(report["test_recording_ids"])
        )
    # Fits involving ordinary recordings remain ordinary; the extreme values
    # can only appear when r3 is itself an outer/inner training recording.
    assert any(value < 100 for value in observed_fit_maxima)
    assert any(value > 1e10 for value in observed_fit_maxima)
    r3_rows = result["scores"].query("recording_id == 'r3'")
    assert all("r3" not in ids for ids in r3_rows["train_recording_ids"])


def test_target_pca_is_outer_training_only_and_reports_coverage():
    records = _recordings(frames=10, target_width=5)
    records["r3"]["targets"][:, 4] *= 1e9
    result = fit_stage4_encoding(
        records, alphas=[1.0], target_pca_components=2
    )
    report = next(
        value
        for value in result["pca_reports"]
        if value["split_kind"] == "primary"
        and value["test_recording_ids"] == ["r3"]
    )

    from sklearn.decomposition import PCA

    training = np.vstack([records[value]["targets"] for value in ("r0", "r1", "r2")])
    expected = PCA(
        n_components=2, svd_solver="full"
    ).fit(training)
    expected_curve = np.cumsum(expected.explained_variance_ratio_)
    assert report["units"] == "target_pca_score"
    assert report["requested_components"] == 2
    assert report["achieved_components"] == 2
    np.testing.assert_allclose(
        report["cumulative_explained_variance_ratio"], expected_curve
    )
    assert report["total_explained_variance_ratio"] == expected_curve[1]
    assert report["train_recording_ids"] == ["r0", "r1", "r2"]
    assert report["test_recording_ids"] == ["r3"]


def test_capacity_matching_is_training_only_common_rank_and_onset_unchanged():
    records = _recordings(frames=10)
    result = fit_stage4_encoding(
        records, alphas=[1.0], capacity_mode=True, sensitivity_folds=None
    )
    report = next(
        value
        for value in result["capacity_reports"]
        if value["split_kind"] == "primary"
        and value["test_recording_ids"] == ["r3"]
    )
    assert report["rank"] == 1  # prosodic/phonetic/word each have one column
    assert report["train_recording_ids"] == ["r0", "r1", "r2"]
    assert report["test_recording_ids"] == ["r3"]
    assert report["onset"] == {
        "input_columns": 1,
        "output_columns": 1,
        "reduced": False,
    }
    assert set(report["families"]) == set(FAMILIES[:4])
    assert all(
        0 <= family_report["explained_variance_coverage"] <= 1 + 1e-12
        for family_report in report["families"].values()
    )
    altered = _recordings(frames=10)
    altered["r3"]["matrix"] *= 1e12
    altered_result = fit_stage4_encoding(
        altered, alphas=[1.0], capacity_mode=True, sensitivity_folds=None
    )
    altered_report = next(
        value
        for value in altered_result["capacity_reports"]
        if value["test_recording_ids"] == ["r3"]
    )
    assert altered_report["families"] == report["families"]


def test_sensitivity_grouped_outer_splits_score_every_held_recording():
    records = _recordings()
    result = fit_stage4_encoding(
        records,
        alphas=[1.0],
        sensitivity_groups={"r0": "a", "r1": "a", "r2": "b", "r3": "b"},
    )
    sensitivity = result["scores"].query("split_kind == 'sensitivity'")
    assert len(sensitivity) == len(records) * len(FAMILIES)
    assert set(sensitivity["recording_id"]) == set(records)
    reports = [
        value
        for value in result["split_reports"]
        if value["split_kind"] == "sensitivity"
    ]
    assert len(reports) == 2
    assert all(
        not set(report["train_recording_ids"]) & set(report["test_recording_ids"])
        for report in reports
    )
    assert set(result["sensitivity_predictions"]) == set(records)


def test_reduced_family_subset_avoids_unrequested_reduced_fits():
    records = _recordings()
    selected = ("prosodic", "phonetic", "word")
    result = fit_stage4_encoding(
        records,
        alphas=[1.0],
        reduced_families=selected,
        sensitivity_folds=None,
    )

    assert set(result["scores"]["family"]) == set(selected)
    assert len(result["scores"]) == len(records) * len(selected)
    assert not result["sensitivity_predictions"]
    for recording_id in records:
        assert set(result["predictions"][recording_id]) == set(selected)


def test_fast_refit_layer_uses_fixed_alphas_without_inner_cv(monkeypatch):
    records = _recordings(count=5)
    from speech_strf import stage4_encoding

    monkeypatch.setattr(
        stage4_encoding,
        "_choose_alpha",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fast refit must not run inner CV")
        ),
    )
    fixed = {
        fold: {"full": 1.0, **{family: 10.0 for family in FAMILIES}}
        for fold in range(5)
    }
    result = fit_stage4_fixed_alpha_grouped(
        records,
        fixed_alphas=fixed,
        lags_seconds=[0.0],
        outer_folds=5,
        target_pca_components=2,
    )

    assert len(result["scores"]) == 5 * len(FAMILIES)
    assert set(result["scores"]["recording_id"]) == set(records)
    assert set(result["scores"]["split_kind"]) == {"primary_fast_grouped"}
    assert set(result["scores"]["full_alpha"]) == {1.0}
    assert set(result["scores"]["reduced_alpha"]) == {10.0}
    assert len(result["pca_reports"]) == 5
    assert all(
        report["inner_cv_repeated"] is False
        for report in result["split_reports"]
    )


def test_sensitivity_reproduces_original_frame_weighted_groupkfold_layout():
    records = _recordings(count=6, frames=8)
    for index, recording_id in enumerate(records):
        keep = 3 + index
        for key in ("matrix", "targets", "times"):
            records[recording_id][key] = records[recording_id][key][:keep]
        records[recording_id]["duration_seconds"] = keep / 50
    result = fit_stage4_encoding(
        records, alphas=[1.0], sensitivity_folds=3
    )
    ids = list(records)
    groups = np.concatenate(
        [np.repeat(recording_id, len(records[recording_id]["matrix"])) for recording_id in ids]
    )
    expected = [
        set(groups[test])
        for _, test in grouped_splits(groups, 3)
    ]
    observed = [
        set(report["test_recording_ids"])
        for report in result["split_reports"]
        if report["split_kind"] == "sensitivity"
    ]
    assert observed == expected


def test_zero_lag_equals_pre_shifted_and_lags_do_not_cross_boundaries():
    records = _recordings(frames=9)
    zero_lag = fit_stage4_encoding(
        records, alphas=[1.0], lags_seconds=[0.0], rate_hz=50
    )
    pre_shifted = fit_stage4_encoding(
        records, alphas=[1.0], pre_shifted=True
    )
    columns = ["full_r2", "reduced_r2", "delta_r2"]
    np.testing.assert_allclose(
        zero_lag["scores"][columns], pre_shifted["scores"][columns]
    )
    for recording_id in records:
        for family in FAMILIES:
            np.testing.assert_allclose(
                zero_lag["predictions"][recording_id][family]["full"],
                pre_shifted["predictions"][recording_id][family]["full"],
            )

    matrix = np.arange(1, 9, dtype=float)[:, None]
    groups = np.array(["a"] * 4 + ["b"] * 4)
    lagged, _ = lagged_design(matrix, groups, ["onset"], [1.0], rate_hz=1)
    assert lagged[4, 0] == 0  # first frame of b must not use last frame of a
    assert lagged[5, 0] == matrix[4, 0]


def test_per_recording_variance_weighted_r2_is_exact():
    truth = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    assert _score(truth, truth) == 1.0
    prediction = np.tile(truth.mean(axis=0), (len(truth), 1))
    assert _score(truth, prediction) == 0.0


def test_positive_and_negative_lag_signs_with_impulse_predictor():
    impulse = np.zeros((7, 1))
    impulse[3, 0] = 1.0
    groups = np.array(["recording"] * len(impulse))
    lagged, _ = lagged_design(
        impulse,
        groups,
        ["acoustic"],
        [-1.0, 0.0, 1.0],
        rate_hz=1.0,
    )
    assert lagged[2, 0] == 1.0  # negative lag uses the future predictor at t + 1
    assert lagged[3, 1] == 1.0
    assert lagged[4, 2] == 1.0  # positive lag uses preceding predictor at t - 1

