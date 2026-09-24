import numpy as np
import yaml

from speech_strf.alignments import Interval
from speech_strf.extract_features import extract_features


def test_feature_matrix_has_shared_rows_and_required_families():
    config = yaml.safe_load(open("configs/features.yaml", encoding="utf-8"))
    sample_rate, duration = 16000, 2.0
    time = np.arange(int(sample_rate * duration)) / sample_rate
    audio = np.sin(2 * np.pi * 180 * time).astype(np.float32)
    intervals = [
        Interval("words", .2, 1.0, "词"),
        Interval("phones", .2, .5, "c"),
    ]
    result = extract_features(audio, sample_rate, duration, intervals, config)
    assert result["matrix"].shape[0] == 100
    assert result["matrix"].shape[1] == len(result["names"]) == len(result["families"])
    assert {"acoustic", "prosodic", "onset", "phonetic", "word"} <= set(
        result["families"]
    )

