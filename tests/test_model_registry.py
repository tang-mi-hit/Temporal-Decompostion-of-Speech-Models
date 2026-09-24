from pathlib import Path

import pytest
import transformers

from speech_strf.adapters import (
    BertTextAdapter,
    FeatureSpeechEncoderAdapter,
    GenericSpeechEncoderAdapter,
    WhisperEncoderAdapter,
    build_adapter,
)
from speech_strf.model_registry import get_model_entry, load_model_registry


REGISTRY = Path(__file__).parents[1] / "configs" / "models.yaml"


def test_planned_registry_identifiers_are_exact():
    defaults, entries = load_model_registry(REGISTRY)
    assert defaults["default_model"] == "hubert_large_reference"
    expected = {
        "hubert_large_reference": "facebook/hubert-large-ls960-ft",
        "hubert_base": "facebook/hubert-base-ls960",
        "wav2vec2_base": "facebook/wav2vec2-base-960h",
        "wav2vec2_large": "facebook/wav2vec2-large-960h-lv60-self",
        "wavlm_base_plus": "microsoft/wavlm-base-plus",
        "wavlm_large": "microsoft/wavlm-large",
        "data2vec_audio_base": "facebook/data2vec-audio-base-960h",
        "xls_r_300m": "facebook/wav2vec2-xls-r-300m",
        "w2v_bert_2": "facebook/w2v-bert-2.0",
        "whisper_medium_encoder": "openai/whisper-medium",
        "mms_1b_all": "facebook/mms-1b-all",
        "bert_base_uncased": "google-bert/bert-base-uncased",
    }
    assert {key: entry.model_id for key, entry in entries.items()} == expected
    assert entries["wav2vec2_base"].loading_class == "Wav2Vec2ForCTC"
    assert entries["wav2vec2_large"].loading_class == "Wav2Vec2ForCTC"
    assert not entries["mms_1b_all"].enabled
    assert entries["hubert_large_reference"].locked_reference


@pytest.mark.parametrize(
    ("key", "adapter_type"),
    [
        ("hubert_large_reference", GenericSpeechEncoderAdapter),
        ("hubert_base", GenericSpeechEncoderAdapter),
        ("wav2vec2_base", GenericSpeechEncoderAdapter),
        ("wav2vec2_large", GenericSpeechEncoderAdapter),
        ("wavlm_base_plus", GenericSpeechEncoderAdapter),
        ("wavlm_large", GenericSpeechEncoderAdapter),
        ("data2vec_audio_base", GenericSpeechEncoderAdapter),
        ("xls_r_300m", GenericSpeechEncoderAdapter),
        ("mms_1b_all", GenericSpeechEncoderAdapter),
        ("w2v_bert_2", FeatureSpeechEncoderAdapter),
        ("whisper_medium_encoder", WhisperEncoderAdapter),
        ("bert_base_uncased", BertTextAdapter),
    ],
)
def test_each_family_resolves_without_loading_weights(key, adapter_type):
    assert isinstance(build_adapter(get_model_entry(REGISTRY, key)), adapter_type)


def test_unknown_model_fails_without_substitution():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model_entry(REGISTRY, "invented_checkpoint")


def test_installed_transformers_exposes_every_registered_loading_class():
    _, entries = load_model_registry(REGISTRY)
    missing = {
        entry.loading_class
        for entry in entries.values()
        if not hasattr(transformers, entry.loading_class)
    }
    assert not missing, f"Transformers {transformers.__version__} lacks {sorted(missing)}"

