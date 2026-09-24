from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def inspect_audio(path: str | Path) -> dict:
    info = sf.info(path)
    return {
        "sample_rate": int(info.samplerate),
        "n_samples": int(info.frames),
        "channels": int(info.channels),
        "duration_seconds": float(info.duration),
        "format": info.format,
        "subtype": info.subtype,
    }


def load_standardized(path: str | Path, target_rate: int) -> tuple[np.ndarray, dict]:
    signal, source_rate = sf.read(path, dtype="float32", always_2d=True)
    metadata = inspect_audio(path)
    actions: list[str] = []
    if signal.shape[1] > 1:
        signal = signal.mean(axis=1, dtype=np.float32)[:, None]
        actions.append("channel_mean_to_mono")
    mono = signal[:, 0]
    if source_rate != target_rate:
        from math import gcd

        divisor = gcd(source_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, source_rate // divisor).astype(
            np.float32
        )
        actions.append(f"polyphase_resample_{source_rate}_to_{target_rate}")
    metadata.update(
        {
            "original_filename": Path(path).name,
            "original_duration_seconds": metadata["duration_seconds"],
            "processed_sample_rate": target_rate,
            "processed_n_samples": len(mono),
            "preprocessing_actions": actions or ["none"],
        }
    )
    return mono, metadata

