from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import librosa
import numpy as np

from .alignments import Interval
from .extract_features import extract_features
from .timebase import resample_continuous


RICH_DESCRIPTOR_NAMES = (
    "spectral_flux_onset_strength",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "zero_crossing_rate",
)
ARCHIVE_KEYS = frozenset({"matrix", "times", "names", "families", "log"})


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_schema_sha256(names: list[str], families: list[str]) -> str:
    """Hash the ordered feature-column contract."""
    return _json_sha256({"names": names, "families": families})


def _finite_differences(values: np.ndarray, step_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < 2:
        zeros = np.zeros_like(values)
        return zeros, zeros.copy()
    first = np.gradient(values, step_seconds, axis=0, edge_order=1)
    second = np.gradient(first, step_seconds, axis=0, edge_order=1)
    return first, second


def _rich_acoustic_features(
    audio: np.ndarray,
    sample_rate: int,
    target_times: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, list[str], dict]:
    rate = float(config.get("analysis_rate_hz", 50))
    if rate != 50.0:
        raise ValueError("Stage 4 rich features require analysis_rate_hz: 50")
    logmel_cfg = config["features"]["log_mel"]
    rich_cfg = config["features"]["rich_acoustic"]
    n_fft = int(logmel_cfg["n_fft"])
    n_mels = int(logmel_cfg["n_mels"])
    hop = max(1, int(round(sample_rate / rate)))

    power = np.abs(
        librosa.stft(
            y=audio,
            n_fft=n_fft,
            hop_length=hop,
            center=False,
        )
    ) ** 2
    mel_power = librosa.feature.melspectrogram(
        S=power,
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        power=2,
    )
    logmel = librosa.power_to_db(mel_power, ref=1.0).T
    frame_times = (np.arange(len(logmel)) * hop + n_fft / 2) / sample_rate

    delta, delta2 = _finite_differences(logmel, hop / sample_rate)
    positive_change = np.maximum(
        np.diff(logmel, axis=0, prepend=logmel[:1]), 0.0
    )
    flux = np.sqrt(np.mean(positive_change**2, axis=1, keepdims=True))

    magnitude = np.sqrt(power)
    descriptor_cfg = rich_cfg["spectral_descriptors"]
    centroid = librosa.feature.spectral_centroid(
        S=magnitude, sr=sample_rate
    ).T
    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude, sr=sample_rate
    ).T
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sample_rate,
        roll_percent=float(descriptor_cfg.get("roll_percent", 0.85)),
    ).T
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=n_fft,
        hop_length=hop,
        center=False,
    ).T

    native = np.column_stack([delta, delta2, flux, centroid, bandwidth, rolloff, zcr])
    native = np.nan_to_num(native, nan=0.0, posinf=0.0, neginf=0.0)
    values = resample_continuous(native, frame_times, target_times)
    names = (
        [f"logmel_delta_{index:02d}" for index in range(n_mels)]
        + [f"logmel_delta2_{index:02d}" for index in range(n_mels)]
        + list(RICH_DESCRIPTOR_NAMES)
    )
    timing = {
        "analysis_rate_hz": rate,
        "hop_length_samples": hop,
        "window_length_samples": n_fft,
        "librosa_center": False,
        "native_frame_center_seconds": "(frame * hop_length + n_fft / 2) / sample_rate",
        "canonical_times_seconds": "frame / analysis_rate_hz",
        "continuous_resampling": "linear_interpolation",
        "temporal_derivative": "numpy_gradient_with_native_hop_seconds",
        "spectral_flux_onset_strength": "rms_positive_logmel_frame_difference",
    }
    return values, names, timing


def extract_stage4_features(
    audio: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
    intervals: list[Interval],
    config: dict,
    *,
    provenance_hashes: Mapping[str, str] | None = None,
) -> dict:
    """Extend the original feature archive with deterministic rich acoustics."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be a mono one-dimensional waveform")

    result = extract_features(audio, sample_rate, duration_seconds, intervals, config)
    rich, rich_names, timing = _rich_acoustic_features(
        audio, sample_rate, result["times"], config
    )

    # Keep the original 40 log-mel columns and envelope in their original order,
    # then place all additional acoustic predictors before the other families.
    try:
        insertion = result["names"].index("broadband_envelope") + 1
    except ValueError as error:
        raise RuntimeError("Base feature schema has no broadband envelope") from error
    matrix = np.column_stack(
        [result["matrix"][:, :insertion], rich, result["matrix"][:, insertion:]]
    ).astype(np.float32)
    names = result["names"][:insertion] + rich_names + result["names"][insertion:]
    families = (
        result["families"][:insertion]
        + ["acoustic"] * len(rich_names)
        + result["families"][insertion:]
    )
    log = dict(result["log"])
    log.update(
        {
            "stage": "stage4_rich",
            "frame_center_convention": timing,
            "feature_schema_sha256": feature_schema_sha256(names, families),
            "provenance_hashes": dict(sorted((provenance_hashes or {}).items())),
        }
    )
    return {
        "matrix": matrix,
        "times": np.asarray(result["times"], dtype=np.float64),
        "names": names,
        "families": families,
        "log": log,
    }


def _integrity_path(archive_path: str | Path) -> Path:
    archive = Path(archive_path)
    return archive.with_name(f"{archive.name}.sha256")


def write_feature_archive_atomic(archive_path: str | Path, result: dict) -> Path:
    """Publish an NPZ with a sidecar commit marker, preserving prior artifacts."""
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _integrity_path(archive)
    existing = [path for path in (archive, sidecar) if path.exists()]
    if existing:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = archive.parent / ".replaced" / f"{archive.name}.{stamp}"
        backup.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, backup / path.name)
    temporary_path: Path | None = None
    sidecar_temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz.tmp", prefix=f".{archive.name}.",
            dir=archive.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                matrix=np.asarray(result["matrix"], dtype=np.float32),
                times=np.asarray(result["times"], dtype=np.float64),
                names=np.asarray(result["names"]),
                families=np.asarray(result["families"]),
                log=json.dumps(result["log"], ensure_ascii=False, sort_keys=True),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        os.replace(temporary_path, archive)
        temporary_path = None

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sha256.tmp", prefix=f".{archive.name}.",
            dir=archive.parent, encoding="ascii", delete=False
        ) as temporary:
            sidecar_temporary_path = Path(temporary.name)
            temporary.write(f"{digest}  {archive.name}\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(sidecar_temporary_path, sidecar)
        sidecar_temporary_path = None
        return archive
    finally:
        for path in (temporary_path, sidecar_temporary_path):
            if path is not None:
                path.unlink(missing_ok=True)


def verify_feature_archive(
    archive_path: str | Path,
    *,
    expected_provenance_hashes: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Validate integrity, schema, dimensions, times, and optional provenance."""
    archive = Path(archive_path)
    sidecar = _integrity_path(archive)
    if not archive.is_file() or not sidecar.is_file():
        return False, "archive_or_integrity_sidecar_missing"
    try:
        fields = sidecar.read_text(encoding="ascii").strip().split()
        if len(fields) != 2 or fields[1] != archive.name or len(fields[0]) != 64:
            return False, "malformed_integrity_sidecar"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != fields[0]:
            return False, "sha256_mismatch"
        with np.load(archive, allow_pickle=False) as stored:
            if set(stored.files) != ARCHIVE_KEYS:
                return False, "archive_schema_mismatch"
            matrix = stored["matrix"]
            times = stored["times"]
            names = stored["names"].tolist()
            families = stored["families"].tolist()
            log = json.loads(stored["log"].item())
        if matrix.ndim != 2 or times.ndim != 1:
            return False, "invalid_array_rank"
        if matrix.shape != (len(times), len(names)) or len(names) != len(families):
            return False, "dimension_mismatch"
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            return False, "non_monotonic_times"
        if len(set(names)) != len(names):
            return False, "duplicate_feature_names"
        if feature_schema_sha256(names, families) != log.get("feature_schema_sha256"):
            return False, "feature_schema_hash_mismatch"
        if expected_provenance_hashes is not None and dict(
            sorted(expected_provenance_hashes.items())
        ) != log.get("provenance_hashes"):
            return False, "provenance_hash_mismatch"
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        KeyError,
        json.JSONDecodeError,
    ):
        return False, "unreadable_or_corrupt_archive"
    return True, "ok"
