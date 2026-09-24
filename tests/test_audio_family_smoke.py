import numpy as np
import pytest

from speech_strf.adapters import canonicalize_states


@pytest.mark.parametrize(
    ("family", "native_rate"),
    [
        ("hubert", 49.95),
        ("wav2vec2", 50.0),
        ("wavlm", 50.0),
        ("data2vec_audio", 50.0),
        ("xls_r", 49.9),
        ("mms", 50.1),
        ("wav2vec2_bert", 50.0),
        ("whisper", 50.0),
    ],
)
def test_audio_family_native_frames_share_identical_canonical_contract(
    family, native_rate
):
    duration = 1.0
    native_times = (np.arange(int(native_rate * duration)) + 0.5) / native_rate
    states = {
        "layer_00_input": np.column_stack(
            [np.sin(native_times), np.cos(native_times)]
        ).astype(np.float32)
    }
    canonical, canonical_times = canonicalize_states(
        states, native_times, duration, canonical_rate_hz=50
    )
    assert family
    assert canonical_times.shape == (50,)
    assert canonical["layer_00_input"].shape == (50, 2)
    assert np.all(np.diff(canonical_times) > 0)

