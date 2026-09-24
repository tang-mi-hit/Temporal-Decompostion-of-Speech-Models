from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from speech_strf.adapters import BertTextAdapter, StimulusRecord
from speech_strf.alignments import Interval
from speech_strf.model_registry import get_model_entry


class FakeEncoding(dict):
    def __init__(self, labels):
        count = len(labels)
        super().__init__(
            input_ids=torch.tensor([[101, *range(10, 10 + count), 102]]),
            attention_mask=torch.ones((1, count + 2), dtype=torch.long),
        )
        self._word_ids = [None, *range(count), None]

    def word_ids(self, batch_index=0):
        return self._word_ids


class FakeFastTokenizer:
    is_fast = True
    unk_token_id = 99
    model_max_length = 8

    def __call__(self, labels, **kwargs):
        return FakeEncoding(labels)


class FakeBert:
    def __call__(self, input_ids, **kwargs):
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
        base = positions[None, :, None].repeat(1, 1, 4)
        return SimpleNamespace(hidden_states=(base, base + 10))


def test_bert_words_are_held_on_audio_grid_without_becoming_predictors():
    registry = Path(__file__).parents[1] / "configs" / "models.yaml"
    entry = get_model_entry(registry, "bert_base_uncased")
    entry.values.update({"device": "cpu", "dtype": "float32", "canonical_rate_hz": 10})
    adapter = BertTextAdapter(entry)
    adapter.processor = FakeFastTokenizer()
    adapter.model = FakeBert()
    record = StimulusRecord(
        "known_sentence",
        duration_seconds=2.0,
        intervals=[
            Interval("words", 0.2, 0.6, "hello"),
            Interval("words", 1.0, 1.4, "world."),
        ],
    )
    result = adapter.extract_hidden_states(record)
    values = result.canonical_states["layer_00_embeddings"]
    assert values.shape == (20, 4)
    assert np.all(values[:2] == 0)
    assert np.all(values[2:6] == 1)
    assert np.all(values[6:10] == 0)
    assert np.all(values[10:14] == 2)
    assert np.all(values[14:] == 0)
    assert result.metadata["alignment_coverage"] == 0.4
    assert result.metadata["context_policy"] == "sentence"
    assert result.metadata["wordpiece_pooling"] == "mean"
    assert result.metadata["bert_predictors_forbidden"] is True
    feature_config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "features.yaml").read_text()
    )
    assert not feature_config["features"]["optional"]["contextual_text_embeddings"][
        "enabled"
    ]


def test_bert_overlength_sentence_fails_without_silent_truncation():
    registry = Path(__file__).parents[1] / "configs" / "models.yaml"
    entry = get_model_entry(registry, "bert_base_uncased")
    entry.values.update({"device": "cpu", "dtype": "float32", "canonical_rate_hz": 10})
    adapter = BertTextAdapter(entry)
    adapter.processor = FakeFastTokenizer()
    adapter.processor.model_max_length = 4
    adapter.model = FakeBert()
    record = StimulusRecord(
        "too_long",
        duration_seconds=1,
        intervals=[
            Interval("words", 0.0, 0.2, "one"),
            Interval("words", 0.2, 0.4, "two"),
            Interval("words", 0.4, 0.6, "three"),
        ],
    )
    with pytest.raises(ValueError, match="requires 5 BERT tokens"):
        adapter.extract_hidden_states(record)

