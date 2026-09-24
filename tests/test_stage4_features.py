from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from speech_strf.alignments import Interval
from speech_strf.stage4_features import (
    RICH_DESCRIPTOR_NAMES,
    extract_stage4_features,
    verify_feature_archive,
    write_feature_archive_atomic,
)


ROOT = Path(__file__).parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/features_stage4_rich.yaml").read_text(encoding="utf-8")
    )


def _fixture() -> tuple[np.ndarray, int, float, list[Interval]]:
    sample_rate, duration = 16000, 1.2
    times = np.arange(round(sample_rate * duration)) / sample_rate
    audio = (
        0.6 * np.sin(2 * np.pi * 220 * times)
        + 0.2 * np.sin(2 * np.pi * 880 * times)
    ).astype(np.float32)
    intervals = [
        Interval("words", 0.1, 0.7, "token"),
        Interval("phones", 0.1, 0.3, "z"),
        Interval("phones", 0.3, 0.7, "a"),
    ]
    return audio, sample_rate, duration, intervals


def test_rich_schema_dimensions_families_and_times_are_deterministic():
    audio, sample_rate, duration, intervals = _fixture()
    config = _config()
    config["phone_categories"] = ["a", "z"]

    first = extract_stage4_features(
        audio, sample_rate, duration, intervals, config,
        provenance_hashes={"input": "abc"},
    )
    second = extract_stage4_features(
        audio, sample_rate, duration, intervals, config,
        provenance_hashes={"input": "abc"},
    )

    expected_rich = (
        [f"logmel_delta_{index:02d}" for index in range(40)]
        + [f"logmel_delta2_{index:02d}" for index in range(40)]
        + list(RICH_DESCRIPTOR_NAMES)
    )
    assert first["names"][:41] == [
        *[f"logmel_{index:02d}" for index in range(40)],
        "broadband_envelope",
    ]
    assert first["names"][41:126] == expected_rich
    assert first["names"].count("f0_hz") == 1
    assert first["names"].count("rms_intensity") == 1
    assert first["names"][-5:] == [
        "phone_category:a",
        "phone_category:z",
        "word_onset",
        "word_boundary",
        "word_duration",
    ]
    assert first["matrix"].shape == (60, 136)
    assert len(first["names"]) == len(first["families"]) == 136
    assert first["families"][:126] == ["acoustic"] * 126
    assert first["families"][first["names"].index("phone_onset")] == "phonetic"
    np.testing.assert_array_equal(first["times"], np.arange(60) / 50)
    np.testing.assert_array_equal(first["matrix"], second["matrix"])
    assert first["names"] == second["names"]
    assert first["log"] == second["log"]


def test_center_false_timing_and_rich_values_are_explicit_and_finite():
    audio, sample_rate, duration, intervals = _fixture()
    config = _config()
    config["phone_categories"] = ["a", "z"]
    result = extract_stage4_features(audio, sample_rate, duration, intervals, config)

    timing = result["log"]["frame_center_convention"]
    assert timing["analysis_rate_hz"] == 50
    assert timing["hop_length_samples"] == 320
    assert timing["window_length_samples"] == 400
    assert timing["librosa_center"] is False
    assert timing["native_frame_center_seconds"].startswith(
        "(frame * hop_length + n_fft / 2)"
    )
    rich = result["matrix"][:, 41:126]
    assert np.isfinite(rich).all()
    assert np.any(rich[:, -5] > 0)  # flux/onset strength
    assert np.all(rich[:, -4] >= 0)  # centroid
    assert np.all((rich[:, -1] >= 0) & (rich[:, -1] <= 1))  # ZCR


def test_atomic_archive_integrity_and_provenance_aware_resume(tmp_path):
    audio, sample_rate, duration, intervals = _fixture()
    config = _config()
    config["phone_categories"] = ["a", "z"]
    provenance = {"audio_sha256": "abc", "config_sha256": "def"}
    result = extract_stage4_features(
        audio, sample_rate, duration, intervals, config,
        provenance_hashes=provenance,
    )
    archive = tmp_path / "recording.npz"

    write_feature_archive_atomic(archive, result)
    assert archive.is_file()
    assert (tmp_path / "recording.npz.sha256").is_file()
    assert verify_feature_archive(
        archive, expected_provenance_hashes=provenance
    ) == (True, "ok")
    assert verify_feature_archive(
        archive, expected_provenance_hashes={"audio_sha256": "changed"}
    ) == (False, "provenance_hash_mismatch")

    with archive.open("ab") as stream:
        stream.write(b"corrupt-partial")
    assert verify_feature_archive(archive) == (False, "sha256_mismatch")


def _write_textgrid(path: Path, phone: str) -> None:
    path.write_text(
        f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.6
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.6
        intervals: size = 1
        intervals [1]:
            xmin = 0.1
            xmax = 0.4
            text = "{phone}"
    item [2]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.6
        intervals: size = 1
        intervals [1]:
            xmin = 0.1
            xmax = 0.4
            text = "word"
''',
        encoding="utf-8",
    )


def test_driver_uses_global_phone_categories_and_repairs_corrupt_resume(tmp_path):
    sample_rate = 16000
    waveform = np.sin(
        2 * np.pi * 300 * np.arange(round(0.6 * sample_rate)) / sample_rate
    ).astype(np.float32)
    rows = []
    for recording_id, phone in (("one", "z"), ("two", "a")):
        audio_path = tmp_path / f"{recording_id}.wav"
        alignment_path = tmp_path / f"{recording_id}.TextGrid"
        sf.write(audio_path, waveform, sample_rate)
        _write_textgrid(alignment_path, phone)
        rows.append(
            {
                "recording_id": recording_id,
                "audio_path": audio_path,
                "alignment_path": alignment_path,
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"valid": True}), encoding="utf-8")
    output = tmp_path / "rich"
    command = [
        sys.executable,
        str(ROOT / "scripts/extract_stage4_rich_features.py"),
        "--config",
        str(ROOT / "configs/features_stage4_rich.yaml"),
        "--manifest",
        str(manifest),
        "--validation-report",
        str(validation),
        "--output",
        str(output),
    ]

    first = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True
    )
    assert first.stdout.count(": extracted") == 2
    schemas = []
    for recording_id in ("one", "two"):
        with np.load(output / f"{recording_id}.npz", allow_pickle=False) as archive:
            schemas.append((archive["names"].tolist(), archive["families"].tolist()))
    assert schemas[0] == schemas[1]
    assert "phone_category:a" in schemas[0][0]
    assert "phone_category:z" in schemas[0][0]
    assert schemas[0][1][schemas[0][0].index("phone_category:a")] == "phonetic"

    with (output / "one.npz").open("ab") as stream:
        stream.write(b"partial")
    resumed = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True
    )
    assert "one: replaced invalid archive (sha256_mismatch)" in resumed.stdout
    assert "two: resumed (integrity verified)" in resumed.stdout
    assert verify_feature_archive(output / "one.npz") == (True, "ok")
    backups = list((output / ".replaced").glob("one.npz.*"))
    assert len(backups) == 1
    assert (backups[0] / "one.npz").is_file()
    assert (backups[0] / "one.npz.sha256").is_file()
    run_manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["global_phone_categories"] == ["a", "z"]
    assert len(run_manifest["global_phone_categories_sha256"]) == 64
