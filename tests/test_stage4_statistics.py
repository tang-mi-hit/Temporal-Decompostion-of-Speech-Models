import numpy as np
import pandas as pd
import pytest

from speech_strf.stage4_statistics import (
    aggregate_layer_effects,
    analyze_stage4_statistics,
    benjamini_hochberg,
    exact_paired_sign_flip,
    paired_differences,
    recording_cluster_bootstrap_ci,
)


def test_exact_sign_flip_enumerates_all_assignments_and_includes_observed():
    result = exact_paired_sign_flip([1.0, 2.0])

    assert result["effect"] == 1.5
    assert result["total_assignments"] == 4
    assert result["extreme_assignments"] == 2
    assert result["p_value"] == 0.5
    assert result["positive_count"] == 2
    assert result["negative_count"] == 0
    assert result["zero_count"] == 0


def test_observed_minus_mean_null_is_the_tested_estimand():
    differences = paired_differences(
        [3.0, 5.0], [[1.0, 3.0], [2.0, 4.0]]
    )
    np.testing.assert_array_equal(differences, [1.0, 2.0])

    result = exact_paired_sign_flip(
        [3.0, 5.0], [[1.0, 3.0], [2.0, 4.0]]
    )
    assert result["estimand"] == "mean_observed_minus_mean_null"
    assert result["effect"] == 1.5
    assert result["p_value"] == 0.5


def test_sign_flip_preserves_positive_negative_and_exact_zero_counts():
    result = exact_paired_sign_flip([2.0, -1.0, 0.0])

    assert result["positive_count"] == 1
    assert result["negative_count"] == 1
    assert result["zero_count"] == 1
    assert result["total_assignments"] == 8


def test_seeded_recording_bootstrap_is_deterministic_and_requires_10000():
    first = recording_cluster_bootstrap_ci(
        [1.0, 2.0, 8.0], n_bootstrap=10_000, seed=91
    )
    second = recording_cluster_bootstrap_ci(
        [1.0, 2.0, 8.0], n_bootstrap=10_000, seed=91
    )

    assert first == second
    assert first == pytest.approx((1.0, 8.0))
    with pytest.raises(ValueError, match="at least 10000"):
        recording_cluster_bootstrap_ci([1.0, 2.0], n_bootstrap=9_999)


def test_benjamini_hochberg_matches_known_five_test_example():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002, 0.5])
    np.testing.assert_allclose(adjusted, [0.025, 0.05, 0.05, 0.01, 0.5])


def _stage4_rows() -> pd.DataFrame:
    rows = []
    effects = {"r1": 1.0, "r2": 2.0, "r3": 3.0}
    durations = {"r1": 1.0, "r2": 1.0, "r3": 4.0}
    for family_number in range(5):
        family = f"family_{family_number}"
        for recording, effect in effects.items():
            # Two layers/checkpoints are averaged to the prespecified effect.
            for layer, offset in (("layer_1", -0.5), ("layer_2", 0.5)):
                rows.append(
                    {
                        "model": "model_a",
                        "recording_id": recording,
                        "family": family,
                        "checkpoint": "fixed_checkpoint",
                        "layer": layer,
                        "effect": effect + offset,
                        "duration_seconds": durations[recording],
                    }
                )
    return pd.DataFrame(rows)


def test_aggregate_layers_before_inference_and_bh_exactly_five_families():
    rows = _stage4_rows()
    aggregate = aggregate_layer_effects(rows)
    assert len(aggregate) == 15
    assert set(aggregate["n_layer_checkpoint_measurements"]) == {2}
    assert aggregate.query(
        "recording_id == 'r2' and family == 'family_3'"
    )["effect"].item() == 2.0

    result = analyze_stage4_statistics(
        rows,
        expected_recordings=["r1", "r2", "r3"],
        families=[f"family_{number}" for number in range(5)],
        n_bootstrap=10_000,
        seed=7,
    )

    assert len(result) == 5
    np.testing.assert_allclose(result["effect"], 2.0)
    np.testing.assert_allclose(result["duration_weighted_effect"], 2.5)
    np.testing.assert_allclose(result["p_value"], 0.25)
    np.testing.assert_allclose(result["q_value"], 0.25)
    assert (result["positive_count"] == 3).all()
    assert (result["negative_count"] == 0).all()
    assert (result["zero_count"] == 0).all()
    assert np.isfinite(
        result[
            [
                "ci_low",
                "ci_high",
                "duration_weighted_ci_low",
                "duration_weighted_ci_high",
            ]
        ].to_numpy()
    ).all()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame[
                ~(
                    (frame["recording_id"] == "r3")
                    & (frame["family"] == "family_4")
                )
            ],
            "missing=\\['r3'\\]",
        ),
        (
            lambda frame: frame.assign(
                effect=np.where(
                    (frame["recording_id"] == "r2")
                    & (frame["family"] == "family_1"),
                    np.nan,
                    frame["effect"],
                )
            ),
            "non-finite layer/checkpoint effects.*r2",
        ),
    ],
)
def test_analysis_diagnoses_incomplete_or_nonfinite_expected_vector(
    mutator, message
):
    with pytest.raises(ValueError, match=message):
        analyze_stage4_statistics(
            mutator(_stage4_rows()),
            expected_recordings=["r1", "r2", "r3"],
            families=[f"family_{number}" for number in range(5)],
            n_bootstrap=10_000,
        )


def test_analysis_rejects_any_family_set_other_than_exactly_five():
    with pytest.raises(ValueError, match="exactly five"):
        analyze_stage4_statistics(
            _stage4_rows(),
            expected_recordings=["r1", "r2", "r3"],
            families=["family_0", "family_1"],
        )
