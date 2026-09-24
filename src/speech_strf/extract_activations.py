from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .audio import load_standardized


def estimate_storage_bytes(
    durations_seconds: list[float],
    frame_rate_hz: float,
    n_layers: int,
    width: int,
    bytes_per_value: int = 4,
) -> int:
    return int(
        sum(durations_seconds)
        * frame_rate_hz
        * n_layers
        * width
        * bytes_per_value
    )


def extract_chunked(
    adapter,
    audio: np.ndarray,
    batch_seconds: float,
    overlap_seconds: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    if batch_seconds <= 0:
        raise ValueError("batch_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= batch_seconds:
        raise ValueError("overlap_seconds must be nonnegative and less than batch_seconds")
    stride, center_offset = adapter.frame_timing_samples()
    sample_rate = adapter.spec.sample_rate_hz
    core_samples = max(
        stride, int(round(batch_seconds * sample_rate / stride)) * stride
    )
    overlap_samples = int(round(overlap_seconds * sample_rate / stride)) * stride
    collected: dict[str, list[np.ndarray]] = {}
    collected_times: list[np.ndarray] = []
    reference_metadata = None
    chunk_count = 0
    extractor = getattr(adapter, "extract_prepared", adapter.extract)
    for core_start in range(0, len(audio), core_samples):
        core_end = min(len(audio), core_start + core_samples)
        context_start = max(0, core_start - overlap_samples)
        context_end = min(len(audio), core_end + overlap_samples)
        states, metadata = extractor(audio[context_start:context_end])
        if reference_metadata is None:
            reference_metadata = metadata
            collected = {name: [] for name in states}
        elif list(states) != list(collected):
            raise RuntimeError("Returned layer names changed between chunks")
        local_count = metadata["frame_count"]
        centers = context_start + center_offset + np.arange(local_count) * stride
        keep = (centers >= core_start) & (centers < core_end) & (centers < len(audio))
        if not np.any(keep):
            raise RuntimeError(f"Chunk {chunk_count} produced no retained activation frames")
        for name, values in states.items():
            collected[name].append(values[keep])
        collected_times.append(centers[keep] / sample_rate)
        chunk_count += 1
    if reference_metadata is None:
        raise ValueError("Cannot extract activations from empty audio")
    frame_times = np.concatenate(collected_times)
    if np.any(np.diff(frame_times) <= 0):
        raise RuntimeError("Chunk stitching produced non-monotonic frame times")
    states = {name: np.concatenate(parts) for name, parts in collected.items()}
    reference_metadata.update(
        {
            "frame_count": len(frame_times),
            "chunk_count": chunk_count,
            "batch_seconds_requested": batch_seconds,
            "batch_seconds_effective": core_samples / sample_rate,
            "overlap_seconds_requested": overlap_seconds,
            "overlap_seconds_effective": overlap_samples / sample_rate,
            "frame_stride_samples": stride,
            "frame_center_offset_samples": center_offset,
            "stitching": "retain_frame_centers_within_nonoverlapping_core_windows",
        }
    )
    return states, frame_times, reference_metadata


def extract_recording(
    adapter,
    audio_path: str,
    recording_id: str,
    store_path: str,
    *,
    batch_seconds: float | None = None,
    overlap_seconds: float = 0.0,
    overwrite: bool = False,
) -> dict:
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    cache_signature = {
        "checkpoint": getattr(adapter.spec, "checkpoint", None),
        "requested_revision": getattr(adapter.spec, "revision", None),
        "sample_rate_hz": adapter.spec.sample_rate_hz,
        "dtype": getattr(adapter, "dtype", None),
        "batch_seconds": batch_seconds,
        "overlap_seconds": overlap_seconds,
        "audio_filename": Path(audio_path).name,
    }
    with h5py.File(store_path, "a") as store:
        if recording_id in store and store[recording_id].attrs.get("complete", False):
            if not overwrite:
                group = store[recording_id]
                existing_model = json.loads(group.attrs["model_metadata_json"])
                if existing_model.get("cache_signature") != cache_signature:
                    raise RuntimeError(
                        f"Cached {recording_id} was created with different extraction "
                        "settings; use --overwrite to recompute it"
                    )
                return {
                    "status": "skipped_complete",
                    "audio": json.loads(group.attrs["audio_metadata_json"]),
                    "model": existing_model,
                    "layers": json.loads(group.attrs["layer_names_json"]),
                }
    audio, audio_metadata = load_standardized(audio_path, adapter.spec.sample_rate_hz)
    if batch_seconds is None:
        states, model_metadata = adapter.extract(audio)
        stride, center_offset = adapter.frame_timing_samples()
        frame_times = (
            center_offset + np.arange(model_metadata["frame_count"]) * stride
        ) / adapter.spec.sample_rate_hz
        model_metadata["chunk_count"] = 1
        model_metadata["stitching"] = "not_chunked"
    else:
        if hasattr(adapter, "prepare_audio"):
            audio, normalization = adapter.prepare_audio(audio)
        else:
            normalization = "adapter_unspecified"
        audio_metadata["model_input_normalization"] = normalization
        states, frame_times, model_metadata = extract_chunked(
            adapter, audio, batch_seconds, overlap_seconds
        )
    frame_count = model_metadata["frame_count"]
    duration = audio_metadata["original_duration_seconds"]
    observed_rate = frame_count / duration
    model_metadata["observed_frame_rate_hz"] = observed_rate
    model_metadata["cache_signature"] = cache_signature
    with h5py.File(store_path, "a") as store:
        temporary_name = f"__incomplete__{recording_id}"
        if temporary_name in store:
            del store[temporary_name]
        group = store.create_group(temporary_name)
        group.attrs["audio_metadata_json"] = json.dumps(audio_metadata)
        group.attrs["model_metadata_json"] = json.dumps(model_metadata)
        group.attrs["layer_names_json"] = json.dumps(list(states))
        group.create_dataset(
            "_frame_times_seconds",
            data=frame_times,
            chunks=(min(4096, len(frame_times)),),
            compression="gzip",
            shuffle=True,
        )
        for name, values in states.items():
            group.create_dataset(
                name,
                data=values,
                chunks=(min(512, len(values)), values.shape[1]),
                compression="gzip",
                shuffle=True,
            )
        group.attrs["complete"] = True
        if recording_id in store:
            del store[recording_id]
        store.move(temporary_name, recording_id)
        store.flush()
    return {
        "status": "written",
        "audio": audio_metadata,
        "model": model_metadata,
        "layers": list(states),
    }

