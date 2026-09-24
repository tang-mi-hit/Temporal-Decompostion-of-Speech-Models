from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def build_manifest(
    audio_root: str | Path,
    alignment_root: str | Path,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    audio_root, alignment_root = Path(audio_root), Path(alignment_root)
    requested = set(sample_ids or [])
    audio_paths = sorted((*audio_root.glob("*.wav"), *audio_root.glob("*.WAV")))
    alignment_paths = sorted(
        (*alignment_root.glob("*.TextGrid"), *alignment_root.glob("*.textgrid"))
    )
    audio_by_id: dict[str, list[Path]] = {}
    alignment_by_id: dict[str, list[Path]] = {}
    for path in audio_paths:
        audio_by_id.setdefault(path.stem, []).append(path)
    for path in alignment_paths:
        alignment_by_id.setdefault(path.stem, []).append(path)
    ids = sorted(requested or (set(audio_by_id) | set(alignment_by_id)))
    rows = []
    for record_id in ids:
        audio = audio_by_id.get(record_id, [])
        alignment = alignment_by_id.get(record_id, [])
        rows.append(
            {
                "recording_id": record_id,
                "story_id": record_id,
                "audio_path": str(audio[0]) if len(audio) == 1 else None,
                "alignment_path": str(alignment[0]) if len(alignment) == 1 else None,
                "audio_matches": len(audio),
                "alignment_matches": len(alignment),
                "audio_filename": audio[0].name if len(audio) == 1 else None,
                "alignment_filename": alignment[0].name if len(alignment) == 1 else None,
            }
        )
    return pd.DataFrame(rows)


def manifest_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_values("recording_id").to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()

