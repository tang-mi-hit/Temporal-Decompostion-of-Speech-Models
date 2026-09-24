from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import hilbert

from .alignments import Interval
from .timebase import events_to_grid, make_time_grid, resample_continuous


def _tier(rows: list[Interval], names: tuple[str, ...]) -> list[Interval]:
    return [row for row in rows if row.tier.lower() in names and row.label.strip()]


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
    intervals: list[Interval],
    config: dict,
) -> dict:
    rate = float(config.get("analysis_rate_hz", 50))
    grid = make_time_grid(duration_seconds, rate)
    hop = max(1, int(round(sample_rate / rate)))
    frames: list[np.ndarray] = []
    names: list[str] = []
    families: list[str] = []
    logmel_cfg = config["features"]["log_mel"]
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=int(logmel_cfg["n_fft"]),
        hop_length=hop,
        n_mels=int(logmel_cfg["n_mels"]),
        power=2,
        center=False,
    )
    mel = librosa.power_to_db(mel, ref=1.0).T
    mel_times = (np.arange(len(mel)) * hop + int(logmel_cfg["n_fft"]) / 2) / sample_rate
    mel = resample_continuous(mel, mel_times, grid.times)
    frames.append(mel)
    names.extend([f"logmel_{index:02d}" for index in range(mel.shape[1])])
    families.extend(["acoustic"] * mel.shape[1])

    envelope = np.abs(hilbert(audio))
    sample_times = np.arange(len(audio)) / sample_rate
    env = resample_continuous(envelope[:, None], sample_times, grid.times)
    frames.append(env)
    names.append("broadband_envelope")
    families.append("acoustic")

    f0_cfg = config["features"]["prosody"]
    f0 = librosa.yin(
        audio,
        fmin=float(f0_cfg["f0_min_hz"]),
        fmax=float(f0_cfg["f0_max_hz"]),
        sr=sample_rate,
        hop_length=hop,
    )
    rms = librosa.feature.rms(y=audio, frame_length=max(2 * hop, 256), hop_length=hop)[0]
    n = min(len(f0), len(rms))
    prosody_times = librosa.frames_to_time(np.arange(n), sr=sample_rate, hop_length=hop)
    prosody = np.column_stack([np.nan_to_num(f0[:n]), (rms[:n] > np.median(rms[:n])).astype(float), rms[:n]])
    frames.append(resample_continuous(prosody, prosody_times, grid.times))
    names.extend(["f0_hz", "voicing", "rms_intensity"])
    families.extend(["prosodic"] * 3)

    onset = np.zeros((len(grid.times), 1))
    if len(onset):
        onset[0, 0] = 1
    frames.append(onset)
    names.append("recording_onset")
    families.append("onset")

    phones = _tier(intervals, ("phone", "phones", "phoneme", "phonemes"))
    phone_labels = config.get("phone_categories", sorted({row.label for row in phones}))
    frames.append(events_to_grid(np.array([row.start for row in phones]), len(grid.times), rate)[:, None])
    names.append("phone_onset")
    families.append("phonetic")
    for label in phone_labels:
        events = [row.start for row in phones if row.label == label]
        frames.append(events_to_grid(np.array(events), len(grid.times), rate)[:, None])
        names.append(f"phone_category:{label}")
        families.append("phonetic")

    words = _tier(intervals, ("word", "words"))
    frames.append(events_to_grid(np.array([row.start for row in words]), len(grid.times), rate)[:, None])
    frames.append(events_to_grid(np.array([row.end for row in words]), len(grid.times), rate)[:, None])
    durations = events_to_grid(
        np.array([row.start for row in words]),
        len(grid.times),
        rate,
        np.array([row.end - row.start for row in words]),
    )
    frames.append(durations[:, None])
    names.extend(["word_onset", "word_boundary", "word_duration"])
    families.extend(["word"] * 3)
    return {
        "matrix": np.column_stack(frames).astype(np.float32),
        "times": grid.times,
        "names": names,
        "families": families,
        "log": {
            "analysis_rate_hz": rate,
            "continuous_resampling": "linear_interpolation",
            "event_resampling": "nearest_frame_accumulation",
            "target_frames": len(grid.times),
        },
    }

