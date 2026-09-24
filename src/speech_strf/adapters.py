from __future__ import annotations

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .alignments import Interval
from .model_registry import HubertAdapter, ModelSpec, RegistryEntry
from .timebase import make_time_grid, resample_continuous


def checkpoint_fingerprint(model_id: str) -> str | None:
    path = Path(model_id)
    if not path.exists():
        return None
    files = []
    if path.is_file():
        files = [path]
    else:
        for pattern in ("*.json", "*.bin", "*.safetensors"):
            files.extend(path.glob(pattern))
    digest = hashlib.sha256()
    for file_path in sorted(set(files)):
        digest.update(file_path.name.encode())
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def configured_extraction_signature(
    entry: RegistryEntry, resolved_revision: str | None = None
) -> dict[str, Any]:
    keys = [
        "model_id",
        "revision",
        "family",
        "input_modality",
        "sample_rate_hz",
        "adapter",
        "loading_class",
        "layers",
        "device",
        "dtype",
        "batch_seconds",
        "chunk_overlap_seconds",
        "canonical_rate_hz",
        "context_policy",
        "sentence_source",
        "sentence_gap_seconds",
        "wordpiece_pooling",
        "outside_word_policy",
        "overlength_sentence_policy",
    ]
    signature = {"model_key": entry.key}
    signature.update({key: entry.values.get(key) for key in keys})
    if signature["loading_class"] == "Wav2Vec2ForCTC":
        # Extraction bypasses the CTC head and uses the same nested encoder as
        # Wav2Vec2Model, so both loading routes have the same activation identity.
        signature["loading_class"] = "Wav2Vec2Model"
    signature["resolved_revision"] = resolved_revision
    signature["local_checkpoint_sha256"] = checkpoint_fingerprint(entry.model_id)
    return signature


@dataclass
class StimulusRecord:
    recording_id: str
    duration_seconds: float
    audio: np.ndarray | None = None
    intervals: list[Interval] | None = None


@dataclass
class ActivationResult:
    native_states: dict[str, np.ndarray]
    native_times: np.ndarray
    canonical_states: dict[str, np.ndarray]
    canonical_times: np.ndarray
    metadata: dict[str, Any]


def canonicalize_states(
    states: dict[str, np.ndarray],
    native_times: np.ndarray,
    duration_seconds: float,
    canonical_rate_hz: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    canonical_times = make_time_grid(duration_seconds, canonical_rate_hz).times
    canonical = {
        name: resample_continuous(values, native_times, canonical_times).astype(
            np.float32, copy=False
        )
        for name, values in states.items()
    }
    return canonical, canonical_times


def save_activation_artifacts(
    store_path: str | Path,
    record: StimulusRecord,
    result: ActivationResult,
    overwrite: bool = False,
) -> str:
    """Atomically save native and canonical activations for one recording."""
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = f"__incomplete__{record.recording_id}"
    with h5py.File(store_path, "a") as store:
        if record.recording_id in store and store[record.recording_id].attrs.get(
            "complete", False
        ):
            if not overwrite:
                existing = json.loads(
                    store[record.recording_id].attrs["metadata_json"]
                )
                if existing.get("extraction_signature") != result.metadata.get(
                    "extraction_signature"
                ):
                    raise RuntimeError(
                        f"Incompatible completed cache for {record.recording_id}; "
                        "use overwrite=True to recompute"
                    )
                return "skipped_complete"
        if temporary in store:
            del store[temporary]
        group = store.create_group(temporary)
        group.attrs["complete"] = False
        group.attrs["metadata_json"] = json.dumps(result.metadata)
        group.attrs["layer_names_json"] = json.dumps(list(result.native_states))
        group.create_dataset("native_timestamps", data=result.native_times)
        group.create_dataset("canonical_timestamps", data=result.canonical_times)
        native = group.create_group("native")
        canonical = group.create_group("canonical")
        for name in result.native_states:
            native_values = result.native_states[name]
            canonical_values = result.canonical_states[name]
            native.create_dataset(
                name,
                data=native_values,
                chunks=(min(512, len(native_values)), native_values.shape[1]),
                compression="gzip",
                shuffle=True,
            )
            canonical.create_dataset(
                name,
                data=canonical_values,
                chunks=(min(512, len(canonical_values)), canonical_values.shape[1]),
                compression="gzip",
                shuffle=True,
            )
        group.attrs["complete"] = True
        if record.recording_id in store:
            del store[record.recording_id]
        store.move(temporary, record.recording_id)
        store.flush()
    return "written"


class ModelAdapter(ABC):
    def __init__(self, entry: RegistryEntry):
        self.entry = entry
        self.model = None
        self.processor = None
        self._last_result: ActivationResult | None = None
        self._configured_signature = configured_extraction_signature(entry)

    def extraction_signature(self, resolved_revision: str | None = None) -> dict:
        signature = dict(self._configured_signature)
        signature["resolved_revision"] = resolved_revision
        return signature

    @abstractmethod
    def load_model(self, local_files_only: bool = True) -> None:
        ...

    @abstractmethod
    def prepare_inputs(self, stimulus_record: StimulusRecord):
        ...

    @abstractmethod
    def extract_hidden_states(self, stimulus_record: StimulusRecord) -> ActivationResult:
        ...

    def get_layer_metadata(self) -> list[dict]:
        if self._last_result is None:
            raise RuntimeError("No extraction has been run")
        return self._last_result.metadata["layers"]

    def get_native_timestamps(self) -> np.ndarray:
        if self._last_result is None:
            raise RuntimeError("No extraction has been run")
        return self._last_result.native_times

    def save_activation_artifacts(
        self,
        path: str | Path,
        record: StimulusRecord,
        overwrite: bool = False,
    ) -> str:
        if self._last_result is None:
            raise RuntimeError("No extraction has been run")
        return save_activation_artifacts(path, record, self._last_result, overwrite)


class GenericSpeechEncoderAdapter(ModelAdapter):
    """Waveform encoder adapter for HuBERT/Wav2Vec2/WavLM/data2vec/XLS-R/MMS."""

    def load_model(self, local_files_only: bool = True) -> None:
        self.backend = HubertAdapter(
            ModelSpec(
                self.entry.model_id,
                self.entry.revision,
                int(self.entry.sample_rate_hz),
            ),
            device=self.entry.device,
            dtype=self.entry.dtype,
            local_files_only=local_files_only,
            loading_class=self.entry.loading_class,
        )
        self.model, self.processor = self.backend.model, self.backend.processor

    def prepare_inputs(self, stimulus_record: StimulusRecord) -> np.ndarray:
        if stimulus_record.audio is None:
            raise ValueError("Audio adapter requires waveform samples")
        prepared, normalization = self.backend.prepare_audio(stimulus_record.audio)
        self._normalization = normalization
        return prepared

    def extract_hidden_states(self, stimulus_record: StimulusRecord) -> ActivationResult:
        from .extract_activations import extract_chunked

        prepared = self.prepare_inputs(stimulus_record)
        states, times, observed = extract_chunked(
            self.backend,
            prepared,
            float(self.entry.batch_seconds),
            float(self.entry.chunk_overlap_seconds),
        )
        states = _select_layers(states, self.entry.layers)
        canonical, canonical_times = canonicalize_states(
            states,
            times,
            stimulus_record.duration_seconds,
            float(self.entry.canonical_rate_hz),
        )
        metadata = self._metadata(states, times, canonical_times, observed)
        metadata["input_normalization"] = self._normalization
        self._last_result = ActivationResult(
            states, times, canonical, canonical_times, metadata
        )
        return self._last_result

    def _metadata(self, states, times, canonical_times, observed) -> dict:
        import torch
        import transformers

        resolved_revision = getattr(self.model.config, "_commit_hash", None)
        return {
            "model_key": self.entry.key,
            "model_id": self.entry.model_id,
            "requested_revision": self.entry.revision,
            "resolved_revision": resolved_revision,
            "extraction_signature": self.extraction_signature(resolved_revision),
            "family": self.entry.family,
            "input_modality": "audio",
            "input_sample_rate_hz": self.entry.sample_rate_hz,
            "native_frame_count": len(times),
            "native_frame_rate_hz": (
                float(1 / np.median(np.diff(times))) if len(times) > 1 else None
            ),
            "canonical_frame_count": len(canonical_times),
            "canonical_rate_hz": self.entry.canonical_rate_hz,
            "canonical_resampling": "linear_interpolation_at_audio_relative_times",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": self.entry.device,
            "dtype": self.entry.dtype,
            "layers": [
                {"name": name, "hidden_size": int(values.shape[1])}
                for name, values in states.items()
            ],
            "extraction": observed,
        }


class FeatureSpeechEncoderAdapter(ModelAdapter):
    """Adapter for waveform frontends returning input_features, including W2V-BERT."""

    def load_model(self, local_files_only: bool = True) -> None:
        import torch
        import transformers

        self.processor = transformers.AutoFeatureExtractor.from_pretrained(
            self.entry.model_id,
            revision=self.entry.revision,
            local_files_only=local_files_only,
        )
        model_class = getattr(transformers, self.entry.loading_class)
        dtype = getattr(torch, self.entry.dtype)
        self.model = model_class.from_pretrained(
            self.entry.model_id,
            revision=self.entry.revision,
            local_files_only=local_files_only,
        ).to(device=self.entry.device, dtype=dtype).eval()

    def prepare_inputs(self, stimulus_record: StimulusRecord):
        if stimulus_record.audio is None:
            raise ValueError("Audio adapter requires waveform samples")
        return self.processor(
            stimulus_record.audio,
            sampling_rate=self.entry.sample_rate_hz,
            return_tensors="pt",
        )

    def extract_hidden_states(self, stimulus_record: StimulusRecord) -> ActivationResult:
        import torch
        import transformers

        if stimulus_record.audio is None:
            raise ValueError("Audio adapter requires waveform samples")
        sample_rate = int(self.entry.sample_rate_hz)
        core_samples = int(round(float(self.entry.batch_seconds) * sample_rate))
        overlap_samples = int(
            round(float(self.entry.chunk_overlap_seconds) * sample_rate)
        )
        collected = None
        collected_times = []
        for core_start in range(0, len(stimulus_record.audio), core_samples):
            core_end = min(len(stimulus_record.audio), core_start + core_samples)
            context_start = max(0, core_start - overlap_samples)
            context_end = min(
                len(stimulus_record.audio), core_end + overlap_samples
            )
            window = StimulusRecord(
                stimulus_record.recording_id,
                (context_end - context_start) / sample_rate,
                audio=stimulus_record.audio[context_start:context_end],
            )
            inputs = self.prepare_inputs(window)
            values = getattr(inputs, "input_features", None)
            if values is None:
                raise RuntimeError(
                    f"{self.entry.key} feature adapter returned no input_features"
                )
            model_inputs = {
                "input_features": values.to(
                    device=self.entry.device,
                    dtype=getattr(torch, self.entry.dtype),
                )
            }
            if "attention_mask" in inputs:
                model_inputs["attention_mask"] = inputs.attention_mask.to(
                    self.entry.device
                )
            with torch.inference_mode():
                output = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            chunk_states = _named_states(output.hidden_states, self.entry.dtype)
            count = len(next(iter(chunk_states.values())))
            context_duration = (context_end - context_start) / sample_rate
            local_times = context_start / sample_rate + _duration_frame_centers(
                context_duration, count
            )
            keep = (local_times >= core_start / sample_rate) & (
                local_times < core_end / sample_rate
            )
            if collected is None:
                collected = {name: [] for name in chunk_states}
            for name, values in chunk_states.items():
                collected[name].append(values[keep])
            collected_times.append(local_times[keep])
        states = _select_layers(
            {name: np.concatenate(parts) for name, parts in collected.items()},
            self.entry.layers,
        )
        times = np.concatenate(collected_times)
        canonical, canonical_times = canonicalize_states(
            states, times, stimulus_record.duration_seconds, self.entry.canonical_rate_hz
        )
        metadata = _simple_metadata(
            self.entry, states, times, canonical_times, torch, transformers
        )
        metadata["resolved_revision"] = getattr(
            getattr(self.model, "config", None), "_commit_hash", None
        )
        metadata["extraction_signature"] = self.extraction_signature(
            metadata["resolved_revision"]
        )
        metadata.update(
            {
                "native_timing_rule": (
                    "actual_per_chunk_output_count_evenly_spans_chunk_duration"
                ),
                "chunk_count": len(collected_times),
            }
        )
        self._last_result = ActivationResult(
            states, times, canonical, canonical_times, metadata
        )
        return self._last_result


class WhisperEncoderAdapter(FeatureSpeechEncoderAdapter):
    """Whisper encoder-only adapter; decoder hidden states are never requested."""

    def extract_hidden_states(self, stimulus_record: StimulusRecord) -> ActivationResult:
        import torch
        import transformers

        if stimulus_record.audio is None:
            raise ValueError("Whisper requires waveform samples")
        sample_rate = int(self.entry.sample_rate_hz)
        core_samples = int(round(float(self.entry.batch_seconds) * sample_rate))
        overlap_samples = int(
            round(float(self.entry.chunk_overlap_seconds) * sample_rate)
        )
        collected = None
        collected_times = []
        encoder = self.model.get_encoder()
        native_rate = None
        for core_start in range(0, len(stimulus_record.audio), core_samples):
            core_end = min(len(stimulus_record.audio), core_start + core_samples)
            context_start = max(0, core_start - overlap_samples)
            context_end = min(
                len(stimulus_record.audio), core_end + overlap_samples
            )
            window = stimulus_record.audio[context_start:context_end]
            inputs = self.processor(
                window, sampling_rate=sample_rate, return_tensors="pt"
            )
            with torch.inference_mode():
                output = encoder(
                    input_features=inputs.input_features.to(
                        device=self.entry.device,
                        dtype=getattr(torch, self.entry.dtype),
                    ),
                    output_hidden_states=True,
                    return_dict=True,
                )
            padded_seconds = float(getattr(self.processor, "chunk_length", 30.0))
            native_rate = output.hidden_states[0].shape[1] / padded_seconds
            context_duration = len(window) / sample_rate
            valid_frames = min(
                output.hidden_states[0].shape[1],
                int(np.ceil(context_duration * native_rate)),
            )
            local_times = (
                context_start / sample_rate + np.arange(valid_frames) / native_rate
            )
            keep = (local_times >= core_start / sample_rate) & (
                local_times < core_end / sample_rate
            )
            if collected is None:
                collected = [[] for _ in output.hidden_states]
            for index, state in enumerate(output.hidden_states):
                collected[index].append(state[0, :valid_frames][keep].detach().cpu())
            collected_times.append(local_times[keep])
        raw = tuple(torch.cat(parts).unsqueeze(0) for parts in collected)
        states = _select_layers(
            _named_states(raw, self.entry.dtype), self.entry.layers
        )
        times = np.concatenate(collected_times)
        canonical, canonical_times = canonicalize_states(
            states, times, stimulus_record.duration_seconds, self.entry.canonical_rate_hz
        )
        metadata = _simple_metadata(
            self.entry, states, times, canonical_times, torch, transformers
        )
        metadata["resolved_revision"] = getattr(
            getattr(self.model, "config", None), "_commit_hash", None
        )
        metadata["extraction_signature"] = self.extraction_signature(
            metadata["resolved_revision"]
        )
        metadata.update(
            {
                "encoder_only": True,
                "native_timing_rule": (
                    "runtime_encoder_rate_with_zero-centered-STFT/conv frame origin"
                ),
                "native_rate_hz_from_runtime": native_rate,
                "chunk_count": len(collected_times),
            }
        )
        self._last_result = ActivationResult(
            states, times, canonical, canonical_times, metadata
        )
        return self._last_result


def segment_words(
    words: list[Interval], gap_seconds: float = 0.5
) -> list[list[Interval]]:
    punctuation = set(".!?。！？；;")
    sentences: list[list[Interval]] = []
    current: list[Interval] = []
    for word in sorted(words, key=lambda item: item.start):
        if current and word.start - current[-1].end >= gap_seconds:
            sentences.append(current)
            current = []
        current.append(word)
        if word.label and word.label[-1] in punctuation:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def hold_word_states_on_grid(
    words: list[Interval],
    word_states: np.ndarray,
    canonical_times: np.ndarray,
    outside_value: float = 0.0,
) -> tuple[np.ndarray, float]:
    if len(words) != len(word_states):
        raise ValueError("Word intervals and representations differ in length")
    output = np.full(
        (len(canonical_times), word_states.shape[1]), outside_value, dtype=word_states.dtype
    )
    covered = np.zeros(len(canonical_times), dtype=bool)
    for word, state in zip(words, word_states):
        mask = (canonical_times >= word.start) & (canonical_times < word.end)
        output[mask] = state
        covered |= mask
    return output, float(covered.mean()) if len(covered) else 0.0


class BertTextAdapter(ModelAdapter):
    def load_model(self, local_files_only: bool = True) -> None:
        import torch
        import transformers

        self.processor = transformers.AutoTokenizer.from_pretrained(
            self.entry.model_id,
            revision=self.entry.revision,
            use_fast=True,
            local_files_only=local_files_only,
        )
        if not self.processor.is_fast:
            raise RuntimeError("BERT alignment requires a fast tokenizer")
        model_class = getattr(transformers, self.entry.loading_class)
        self.model = model_class.from_pretrained(
            self.entry.model_id,
            revision=self.entry.revision,
            local_files_only=local_files_only,
        ).to(
            device=self.entry.device,
            dtype=getattr(torch, self.entry.dtype),
        ).eval()

    def prepare_inputs(self, stimulus_record: StimulusRecord):
        words = [
            row
            for row in (stimulus_record.intervals or [])
            if row.tier.lower() in {"word", "words"} and row.label.strip()
        ]
        return segment_words(words, float(self.entry.sentence_gap_seconds))

    def extract_hidden_states(self, stimulus_record: StimulusRecord) -> ActivationResult:
        import torch
        import transformers

        sentences = self.prepare_inputs(stimulus_record)
        words = [word for sentence in sentences for word in sentence]
        if not words:
            raise ValueError(f"{stimulus_record.recording_id} has no labeled word intervals")
        layer_words: list[list[np.ndarray]] | None = None
        unknown_tokens = 0
        total_tokens = 0
        with torch.inference_mode():
            for sentence in sentences:
                labels = [word.label for word in sentence]
                encoded = self.processor(
                    labels,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=False,
                )
                token_count = int(encoded["input_ids"].shape[1])
                tokenizer_limit = int(
                    getattr(self.processor, "model_max_length", 10**9)
                )
                model_limit = int(
                    getattr(
                        getattr(self.model, "config", None),
                        "max_position_embeddings",
                        tokenizer_limit,
                    )
                )
                limit = min(tokenizer_limit, model_limit)
                if token_count > limit:
                    raise ValueError(
                        f"Sentence requires {token_count} BERT tokens but the model "
                        f"limit is {limit}; overlength_sentence_policy=error"
                    )
                word_ids = encoded.word_ids(batch_index=0)
                model_inputs = {
                    key: value.to(self.entry.device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
                output = self.model(
                    **model_inputs, output_hidden_states=True, return_dict=True
                )
                if layer_words is None:
                    layer_words = [[] for _ in output.hidden_states]
                ids = encoded["input_ids"][0]
                unknown_tokens += int((ids == self.processor.unk_token_id).sum())
                total_tokens += int(len(ids) - 2)
                for layer_index, state in enumerate(output.hidden_states):
                    state = state[0].detach().cpu().numpy()
                    for word_index in range(len(sentence)):
                        positions = [i for i, owner in enumerate(word_ids) if owner == word_index]
                        if not positions:
                            raise RuntimeError(
                                f"Tokenizer produced no wordpieces for {labels[word_index]!r}"
                            )
                        layer_words[layer_index].append(state[positions].mean(axis=0))
        names = ["layer_00_embeddings"] + [
            f"layer_{index:02d}_transformer" for index in range(1, len(layer_words))
        ]
        native_states = {
            name: np.asarray(values, dtype=self.entry.dtype)
            for name, values in zip(names, layer_words)
        }
        native_states = _select_layers(native_states, self.entry.layers)
        native_times = np.asarray([(word.start + word.end) / 2 for word in words])
        canonical_times = make_time_grid(
            stimulus_record.duration_seconds, self.entry.canonical_rate_hz
        ).times
        canonical_states = {}
        coverages = []
        for name, values in native_states.items():
            canonical_states[name], coverage = hold_word_states_on_grid(
                words, values, canonical_times, outside_value=0.0
            )
            canonical_states[name] = canonical_states[name].astype(
                np.float32, copy=False
            )
            coverages.append(coverage)
        metadata = _simple_metadata(
            self.entry,
            native_states,
            native_times,
            canonical_times,
            torch,
            transformers,
        )
        metadata["resolved_revision"] = getattr(
            getattr(self.model, "config", None), "_commit_hash", None
        )
        metadata["extraction_signature"] = self.extraction_signature(
            metadata["resolved_revision"]
        )
        metadata.update(
            {
                "baseline_label": "text-only baseline",
                "context_policy": self.entry.context_policy,
                "sentence_source": self.entry.sentence_source,
                "sentence_gap_seconds": self.entry.sentence_gap_seconds,
                "wordpiece_pooling": self.entry.wordpiece_pooling,
                "outside_word_policy": self.entry.outside_word_policy,
                "alignment_coverage": coverages[0],
                "unknown_token_fraction": unknown_tokens / max(total_tokens, 1),
                "bert_predictors_forbidden": True,
            }
        )
        self._last_result = ActivationResult(
            native_states,
            native_times,
            canonical_states,
            canonical_times,
            metadata,
        )
        return self._last_result


def build_adapter(entry: RegistryEntry) -> ModelAdapter:
    adapters = {
        "generic_speech": GenericSpeechEncoderAdapter,
        "feature_speech": FeatureSpeechEncoderAdapter,
        "whisper_encoder": WhisperEncoderAdapter,
        "bert_text": BertTextAdapter,
    }
    try:
        return adapters[entry.adapter](entry)
    except KeyError as exc:
        raise ValueError(f"Unsupported adapter {entry.adapter!r}") from exc


def _named_states(hidden_states, dtype: str) -> dict[str, np.ndarray]:
    arrays = [
        state[0].detach().cpu().numpy().astype(dtype, copy=False)
        for state in hidden_states
    ]
    names = ["layer_00_input"] + [
        f"layer_{index:02d}_transformer" for index in range(1, len(arrays))
    ]
    return dict(zip(names, arrays))


def _select_layers(
    states: dict[str, np.ndarray], selection: str | list[str]
) -> dict[str, np.ndarray]:
    if selection == "all":
        return states
    missing = set(selection) - set(states)
    if missing:
        raise ValueError(f"Requested layers were not returned: {sorted(missing)}")
    return {name: states[name] for name in selection}


def _duration_frame_centers(duration: float, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("Model returned no frames")
    return (np.arange(count) + 0.5) * duration / count


def _simple_metadata(entry, states, times, canonical_times, torch, transformers):
    return {
        "model_key": entry.key,
        "model_id": entry.model_id,
        "requested_revision": entry.revision,
        "resolved_revision": None,
        "family": entry.family,
        "input_modality": entry.input_modality,
        "input_sample_rate_hz": entry.sample_rate_hz,
        "native_frame_count": len(times),
        "native_frame_rate_hz": (
            float(1 / np.median(np.diff(times))) if len(times) > 1 else None
        ),
        "canonical_frame_count": len(canonical_times),
        "canonical_rate_hz": entry.canonical_rate_hz,
        "canonical_resampling": (
            "word_interval_hold" if entry.input_modality == "text"
            else "linear_interpolation_at_audio_relative_times"
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": entry.device,
        "dtype": entry.dtype,
        "layers": [
            {"name": name, "hidden_size": int(values.shape[1])}
            for name, values in states.items()
        ],
    }
