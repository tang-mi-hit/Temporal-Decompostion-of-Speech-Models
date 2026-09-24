import numpy as np
import pandas as pd
import soundfile as sf

from speech_strf.validate_inputs import validate_manifest


def test_tolerated_empty_overhang_keeps_manifest_valid(tmp_path):
    audio = tmp_path / "sample.wav"
    alignment = tmp_path / "sample.TextGrid"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    alignment.write_text(
        '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1.026
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.026
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.026
            text = ""
''',
        encoding="utf-8",
    )
    manifest = pd.DataFrame(
        [
            {
                "recording_id": "sample",
                "audio_path": str(audio),
                "alignment_path": str(alignment),
                "audio_matches": 1,
                "alignment_matches": 1,
                "audio_filename": audio.name,
                "alignment_filename": alignment.name,
            }
        ]
    )
    report = validate_manifest(manifest, empty_end_tolerance_seconds=0.03)
    assert report["valid"]
    assert report["issue_count"] == 0
    assert report["warning_count"] == 1

