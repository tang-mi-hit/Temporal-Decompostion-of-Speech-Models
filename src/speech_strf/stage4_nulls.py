"""Deterministic recording-local circular-shift null predictors."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse


SUBSTANTIVE_FAMILIES = ("prosodic", "phonetic", "word")
TARGET_NULL_COUNT = 100
MINIMUM_DRY_RUN_NULL_COUNT = 20


class ShortRecordingError(ValueError):
    """Raised when one or more recordings have no permissible circular shift."""

    def __init__(self, diagnostics: Sequence[Mapping[str, Any]]) -> None:
        self.diagnostics = [dict(value) for value in diagnostics]
        ids = ", ".join(value["recording_id"] for value in self.diagnostics)
        super().__init__(
            "No permissible recording-local circular shift for: "
            f"{ids}. See .diagnostics for required and available durations."
        )


def validate_null_count(null_count: int, *, dry_run: bool = False) -> int:
    """Validate the fixed production target or the dry-run minimum."""
    if isinstance(null_count, bool) or not isinstance(null_count, (int, np.integer)):
        raise TypeError("null_count must be an integer")
    null_count = int(null_count)
    minimum = MINIMUM_DRY_RUN_NULL_COUNT if dry_run else TARGET_NULL_COUNT
    if dry_run:
        if null_count < minimum:
            raise ValueError(
                f"dry-run null_count must be at least {MINIMUM_DRY_RUN_NULL_COUNT}"
            )
    elif null_count != TARGET_NULL_COUNT:
        raise ValueError(
            f"production null_count must equal the target {TARGET_NULL_COUNT}"
        )
    return null_count


def _matrix_and_families(
    recording_id: str, recording: Mapping[str, Any]
) -> tuple[str, np.ndarray, list[str]]:
    matrix_key = next(
        (
            key
            for key in ("matrix", "features", "feature_matrix")
            if recording.get(key) is not None
        ),
        None,
    )
    family_key = next(
        (
            key
            for key in ("families", "feature_families")
            if recording.get(key) is not None
        ),
        None,
    )
    if matrix_key is None or family_key is None:
        raise ValueError(f"{recording_id}: matrix and families are required")
    matrix = recording[matrix_key]
    if not sparse.issparse(matrix):
        matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"{recording_id}: matrix must be two-dimensional")
    families = np.asarray(recording[family_key]).astype(str).tolist()
    if matrix.shape[1] != len(families):
        raise ValueError(f"{recording_id}: matrix columns and families disagree")
    missing = [family for family in SUBSTANTIVE_FAMILIES if family not in families]
    if missing:
        raise ValueError(
            f"{recording_id}: missing substantive families {missing}"
        )
    return matrix_key, matrix, families


def _minimum_shift_frames(
    rate_hz: float, minimum_zero_seconds: float, max_abs_lag_seconds: float
) -> int:
    # A shift may equal the minimum-zero threshold, but must lie strictly
    # outside the largest tested lag.
    zero_frames = int(np.ceil(minimum_zero_seconds * rate_hz))
    lag_frames = int(np.floor(max_abs_lag_seconds * rate_hz)) + 1
    return max(1, zero_frames, lag_frames)


def _derived_seed(
    seed: int, null_index: int, recording_id: str, family: str
) -> int:
    payload = (
        f"speech-strf-stage4-null-v1\0{seed}\0{null_index}\0"
        f"{recording_id}\0{family}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _roll_columns(
    matrix: np.ndarray, columns: np.ndarray, offset_frames: int
) -> np.ndarray:
    if sparse.issparse(matrix):
        original_format = matrix.format
        shifted = matrix.tocoo(copy=True)
        selected = columns[shifted.col]
        shifted.row[selected] = (
            shifted.row[selected] + offset_frames
        ) % matrix.shape[0]
        return shifted.asformat(original_format)
    shifted = np.array(matrix, copy=True)
    shifted[:, columns] = np.roll(
        shifted[:, columns], offset_frames, axis=0
    )
    return shifted


def circular_shift_null(
    recordings: Mapping[str, Mapping[str, Any]],
    *,
    null_index: int,
    seed: int = 0,
    rate_hz: float = 50.0,
    lags_seconds: Sequence[float] = (0.0,),
    max_abs_lag_seconds: float | None = None,
    minimum_zero_seconds: float = 2.0,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return one deterministic null draw and its shift manifest.

    Each substantive family is shifted as a complete column block, using a
    separately derived seed for every recording/family pair. All wrapping is
    confined to rows of the same recording.
    """
    if isinstance(null_index, bool) or not isinstance(
        null_index, (int, np.integer)
    ):
        raise TypeError("null_index must be a nonnegative integer")
    null_index = int(null_index)
    if null_index < 0:
        raise ValueError("null_index must be a nonnegative integer")
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    if not np.isfinite(minimum_zero_seconds) or minimum_zero_seconds < 2.0:
        raise ValueError("minimum_zero_seconds must be finite and at least 2")
    lag_values = np.asarray(lags_seconds, dtype=float)
    if lag_values.ndim != 1 or lag_values.size == 0 or not np.isfinite(
        lag_values
    ).all():
        raise ValueError("lags_seconds must be a non-empty finite sequence")
    tested_max_lag = float(np.max(np.abs(lag_values)))
    if max_abs_lag_seconds is not None:
        if (
            not np.isfinite(max_abs_lag_seconds)
            or max_abs_lag_seconds < 0
        ):
            raise ValueError("max_abs_lag_seconds must be finite and nonnegative")
        tested_max_lag = max(tested_max_lag, float(max_abs_lag_seconds))

    minimum_frames = _minimum_shift_frames(
        float(rate_hz), float(minimum_zero_seconds), tested_max_lag
    )
    prepared: dict[str, tuple[str, np.ndarray, list[str]]] = {}
    diagnostics = []
    for recording_id, recording in sorted(recordings.items(), key=lambda x: str(x[0])):
        identity = str(recording_id)
        prepared[identity] = _matrix_and_families(identity, recording)
        frames = int(prepared[identity][1].shape[0])
        maximum_frames = frames - minimum_frames
        if maximum_frames < minimum_frames:
            diagnostics.append(
                {
                    "recording_id": identity,
                    "frames": frames,
                    "duration_seconds": frames / rate_hz,
                    "minimum_offset_frames": minimum_frames,
                    "minimum_offset_seconds": minimum_frames / rate_hz,
                    "required_minimum_frames": 2 * minimum_frames,
                    "required_minimum_seconds": 2 * minimum_frames / rate_hz,
                    "reason": "no_offset_outside_zero_lag_exclusion",
                }
            )
    if diagnostics:
        raise ShortRecordingError(diagnostics)

    shifted_recordings: dict[str, dict[str, Any]] = {}
    recording_manifest: dict[str, dict[str, Any]] = {}
    source_by_id = {str(key): value for key, value in recordings.items()}
    for recording_id in sorted(prepared):
        matrix_key, matrix, families = prepared[recording_id]
        frames = matrix.shape[0]
        maximum_frames = frames - minimum_frames
        shifted_matrix = matrix.copy()
        family_manifest: dict[str, dict[str, Any]] = {}
        family_array = np.asarray(families)
        for family in SUBSTANTIVE_FAMILIES:
            derived_seed = _derived_seed(seed, null_index, recording_id, family)
            rng = np.random.default_rng(derived_seed)
            offset = int(rng.integers(minimum_frames, maximum_frames + 1))
            columns = family_array == family
            shifted_matrix = _roll_columns(shifted_matrix, columns, offset)
            family_manifest[family] = {
                "offset_frames": offset,
                "offset_seconds": offset / rate_hz,
                "circular_distance_frames": min(offset, frames - offset),
                "circular_distance_seconds": min(offset, frames - offset) / rate_hz,
                "permitted_offset_frames": [minimum_frames, maximum_frames],
                "permitted_offset_seconds": [
                    minimum_frames / rate_hz,
                    maximum_frames / rate_hz,
                ],
                "derived_seed": derived_seed,
                "columns": int(columns.sum()),
            }
        copied = copy.deepcopy(dict(source_by_id[recording_id]))
        copied[matrix_key] = shifted_matrix
        shifted_recordings[recording_id] = copied
        recording_manifest[recording_id] = {
            "frames": int(frames),
            "duration_seconds": frames / rate_hz,
            "families": family_manifest,
        }

    manifest = {
        "null_index": null_index,
        "seed": int(seed),
        "rate_hz": float(rate_hz),
        "minimum_zero_seconds": float(minimum_zero_seconds),
        "max_abs_lag_seconds": tested_max_lag,
        "minimum_offset_frames": minimum_frames,
        "recordings": recording_manifest,
    }
    return shifted_recordings, manifest


def generate_stage4_nulls(
    recordings: Mapping[str, Mapping[str, Any]],
    *,
    null_count: int = TARGET_NULL_COUNT,
    dry_run: bool = False,
    seed: int = 0,
    rate_hz: float = 50.0,
    lags_seconds: Sequence[float] = (0.0,),
    max_abs_lag_seconds: float | None = None,
    minimum_zero_seconds: float = 2.0,
) -> list[tuple[dict[str, dict[str, Any]], dict[str, Any]]]:
    """Generate a validated production or dry-run null ensemble."""
    count = validate_null_count(null_count, dry_run=dry_run)
    return [
        circular_shift_null(
            recordings,
            null_index=null_index,
            seed=seed,
            rate_hz=rate_hz,
            lags_seconds=lags_seconds,
            max_abs_lag_seconds=max_abs_lag_seconds,
            minimum_zero_seconds=minimum_zero_seconds,
        )
        for null_index in range(count)
    ]


make_stage4_null = circular_shift_null

