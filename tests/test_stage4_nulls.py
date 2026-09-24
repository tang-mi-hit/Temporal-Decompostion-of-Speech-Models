from __future__ import annotations

import numpy as np
import pytest

from speech_strf.stage4_nulls import (
    ShortRecordingError,
    circular_shift_null,
    validate_null_count,
)


FAMILIES = [
    "acoustic",
    "prosodic",
    "prosodic",
    "phonetic",
    "phonetic",
    "word",
    "word",
    "onset",
]


def _recordings(frames: int = 600):
    recordings = {}
    for recording_index, recording_id in enumerate(("first", "second")):
        time = np.arange(frames)
        matrix = np.column_stack(
            [
                10000 * recording_index + time,
                (time % 13 == 0).astype(float),
                3 * (time % 13 == 0),
                (time % 17 == 2).astype(float),
                5 + 2 * (time % 17 == 2),
                time % 23,
                100 + time % 23,
                time == 0,
            ]
        )
        recordings[recording_id] = {
            "matrix": matrix,
            "families": FAMILIES,
            "times": time / 50,
            "targets": np.full((frames, 2), recording_index),
        }
    return recordings


def test_shifts_wrap_only_inside_each_recording():
    source = _recordings()
    shifted, manifest = circular_shift_null(
        source, null_index=3, seed=91, lags_seconds=(-0.5, 0.0, 0.5)
    )

    for recording_id, original in source.items():
        for family in ("prosodic", "phonetic", "word"):
            columns = np.asarray(FAMILIES) == family
            offset = manifest["recordings"][recording_id]["families"][family][
                "offset_frames"
            ]
            np.testing.assert_array_equal(
                shifted[recording_id]["matrix"][:, columns],
                np.roll(original["matrix"][:, columns], offset, axis=0),
            )
        # Values from one recording can never enter the other, and untouched
        # families remain byte-for-byte unchanged.
        for family in ("acoustic", "onset"):
            columns = np.asarray(FAMILIES) == family
            np.testing.assert_array_equal(
                shifted[recording_id]["matrix"][:, columns],
                original["matrix"][:, columns],
            )
    assert not np.shares_memory(
        shifted["first"]["matrix"], source["first"]["matrix"]
    )


def test_whole_family_blocks_preserve_sparsity_and_column_relations():
    shifted, _ = circular_shift_null(_recordings(), null_index=0, seed=5)
    matrix = shifted["first"]["matrix"]

    assert np.count_nonzero(matrix[:, 1]) == np.count_nonzero(matrix[:, 2])
    np.testing.assert_array_equal(matrix[:, 2], 3 * matrix[:, 1])
    np.testing.assert_array_equal(matrix[:, 4], 5 + 2 * matrix[:, 3])
    np.testing.assert_array_equal(matrix[:, 6], 100 + matrix[:, 5])


def test_offsets_and_null_index_seeds_are_deterministic_and_distinct():
    source = _recordings()
    first_shifted, first = circular_shift_null(
        source, null_index=7, seed=1234
    )
    second_shifted, second = circular_shift_null(
        source, null_index=7, seed=1234
    )
    _, next_null = circular_shift_null(source, null_index=8, seed=1234)

    assert first == second
    for recording_id in source:
        np.testing.assert_array_equal(
            first_shifted[recording_id]["matrix"],
            second_shifted[recording_id]["matrix"],
        )
        seeds = [
            details["derived_seed"]
            for details in first["recordings"][recording_id]["families"].values()
        ]
        assert len(seeds) == len(set(seeds))
        assert all(
            first["recordings"][recording_id]["families"][family]["derived_seed"]
            != next_null["recordings"][recording_id]["families"][family][
                "derived_seed"
            ]
            for family in ("prosodic", "phonetic", "word")
        )


def test_offsets_exclude_zero_lag_and_equivalent_near_full_wrap():
    _, manifest = circular_shift_null(
        _recordings(frames=600),
        null_index=4,
        seed=8,
        rate_hz=50,
        lags_seconds=(-2.4, 0.0, 2.4),
    )

    # Strictly outside 2.4 seconds means at least 121 frames in either
    # circular direction.
    assert manifest["minimum_offset_frames"] == 121
    for recording in manifest["recordings"].values():
        for details in recording["families"].values():
            assert details["permitted_offset_frames"] == [121, 479]
            assert 121 <= details["offset_frames"] <= 479
            assert details["circular_distance_frames"] >= 121


def test_short_recording_reports_diagnostic_instead_of_fabricating_shift():
    source = _recordings(frames=199)
    with pytest.raises(ShortRecordingError) as caught:
        circular_shift_null(source, null_index=0, rate_hz=50)

    diagnostics = caught.value.diagnostics
    assert [value["recording_id"] for value in diagnostics] == [
        "first",
        "second",
    ]
    assert all(value["frames"] == 199 for value in diagnostics)
    assert all(value["required_minimum_frames"] == 200 for value in diagnostics)
    np.testing.assert_array_equal(
        source["first"]["matrix"], _recordings(frames=199)["first"]["matrix"]
    )


def test_production_target_and_dry_run_minimum_are_validated():
    assert validate_null_count(100) == 100
    assert validate_null_count(20, dry_run=True) == 20
    with pytest.raises(ValueError, match="at least 20"):
        validate_null_count(19, dry_run=True)
    with pytest.raises(ValueError, match="target 100"):
        validate_null_count(99)

