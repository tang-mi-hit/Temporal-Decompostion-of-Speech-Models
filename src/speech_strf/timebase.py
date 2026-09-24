from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    times: np.ndarray
    rate_hz: float
    duration_seconds: float


def make_time_grid(duration_seconds: float, rate_hz: float = 50.0) -> TimeGrid:
    if duration_seconds < 0 or rate_hz <= 0:
        raise ValueError("duration_seconds must be nonnegative and rate_hz positive")
    n_frames = int(np.floor(duration_seconds * rate_hz + 1e-9))
    return TimeGrid(np.arange(n_frames, dtype=float) / rate_hz, rate_hz, duration_seconds)


def nearest_frame(times: np.ndarray, rate_hz: float, n_frames: int) -> np.ndarray:
    indices = np.rint(np.asarray(times, dtype=float) * rate_hz).astype(int)
    return np.clip(indices, 0, max(0, n_frames - 1))


def resample_continuous(
    values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    values = np.asarray(values)
    source_times = np.asarray(source_times)
    if values.shape[0] != source_times.size:
        raise ValueError("values and source_times lengths differ")
    if source_times.size == 0:
        return np.zeros((target_times.size,) + values.shape[1:], dtype=float)
    if np.any(np.diff(source_times) <= 0):
        raise ValueError("source_times must be strictly increasing")
    flat = values.reshape(values.shape[0], -1)
    out = np.column_stack(
        [np.interp(target_times, source_times, flat[:, j]) for j in range(flat.shape[1])]
    )
    return out.reshape((target_times.size,) + values.shape[1:])


def events_to_grid(
    event_times: np.ndarray, n_frames: int, rate_hz: float, weights: np.ndarray | None = None
) -> np.ndarray:
    result = np.zeros(n_frames, dtype=float)
    if len(event_times) == 0 or n_frames == 0:
        return result
    indices = nearest_frame(np.asarray(event_times), rate_hz, n_frames)
    np.add.at(result, indices, 1.0 if weights is None else np.asarray(weights, dtype=float))
    return result

