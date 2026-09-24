from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REQUIRED_MODEL_FIELDS = {
    "model_id",
    "revision",
    "family",
    "input_modality",
    "sample_rate_hz",
    "adapter",
    "loading_class",
    "layers",
    "enabled",
    "license_notes",
    "limitations",
}
SUPPORTED_ADAPTERS = {
    "generic_speech",
    "feature_speech",
    "whisper_encoder",
    "bert_text",
}


@dataclass(frozen=True)
class RegistryEntry:
    key: str
    values: dict[str, Any]

    def __getattr__(self, name: str):
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, **self.values}


def validate_registry(config: dict) -> list[str]:
    errors: list[str] = []
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        return ["models must be a non-empty mapping"]
    if config.get("default_model") not in models:
        errors.append("default_model must name a registry entry")
    for key, values in models.items():
        if not key or key.lower() != key or " " in key:
            errors.append(f"{key!r}: stable key must be lowercase and contain no spaces")
        missing = REQUIRED_MODEL_FIELDS - set(values)
        if missing:
            errors.append(f"{key}: missing fields {sorted(missing)}")
            continue
        if values["input_modality"] not in {"audio", "text"}:
            errors.append(f"{key}: input_modality must be audio or text")
        if values["input_modality"] == "audio" and not values["sample_rate_hz"]:
            errors.append(f"{key}: audio models require sample_rate_hz")
        if values["input_modality"] == "text" and values["sample_rate_hz"] is not None:
            errors.append(f"{key}: text models must use null sample_rate_hz")
        if values["adapter"] not in SUPPORTED_ADAPTERS:
            errors.append(f"{key}: unsupported adapter {values['adapter']!r}")
        if values.get("locked_reference") and not values.get("output_dir"):
            errors.append(f"{key}: locked references require output_dir")
        if values["adapter"] == "whisper_encoder":
            batch = config.get("runtime_defaults", {}).get("batch_seconds")
            if not batch or batch > 30:
                errors.append(f"{key}: Whisper batch_seconds must be in (0, 30]")
        if values["adapter"] == "bert_text":
            required = {
                "context_policy",
                "sentence_source",
                "sentence_gap_seconds",
                "wordpiece_pooling",
                "outside_word_policy",
                "overlength_sentence_policy",
            }
            if missing_text := required - set(values):
                errors.append(f"{key}: missing BERT fields {sorted(missing_text)}")
        if values["layers"] != "all" and not isinstance(values["layers"], list):
            errors.append(f"{key}: layers must be 'all' or a list of layer names")
    return errors


def load_model_registry(path: str | Path) -> tuple[dict, dict[str, RegistryEntry]]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    errors = validate_registry(config)
    if errors:
        raise ValueError("Invalid model registry:\n- " + "\n- ".join(errors))
    defaults = dict(config.get("runtime_defaults", {}))
    defaults["default_model"] = config["default_model"]
    entries = {
        key: RegistryEntry(key, {**defaults, **values})
        for key, values in config["models"].items()
    }
    return defaults, entries


def get_model_entry(path: str | Path, key: str | None = None) -> RegistryEntry:
    defaults, entries = load_model_registry(path)
    selected = key or defaults["default_model"]
    if selected not in entries:
        raise KeyError(f"Unknown model {selected!r}; available: {sorted(entries)}")
    return entries[selected]


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    revision: str
    sample_rate_hz: int


class HubertAdapter:
    """Hugging Face HuBERT adapter with explicit device and precision controls."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "cpu",
        dtype: str = "float32",
        local_files_only: bool = False,
        loading_class: str = "HubertModel",
        processor=None,
        model=None,
    ) -> None:
        import torch

        if dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be float16 or float32")
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("device must be cpu, cuda, or a CUDA device such as cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if not device.startswith("cuda") and dtype == "float16":
            raise ValueError("float16 inference is supported only on CUDA")
        self.spec, self.device, self.dtype = spec, device, dtype
        self.torch_dtype = getattr(torch, dtype)
        self.numpy_dtype = np.dtype(dtype)
        self.loading_info = {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
        }
        if processor is None or model is None:
            import transformers

            processor = transformers.AutoFeatureExtractor.from_pretrained(
                spec.checkpoint,
                revision=spec.revision,
                local_files_only=local_files_only,
            )
            try:
                model_class = getattr(transformers, loading_class)
            except AttributeError as exc:
                raise ValueError(f"Unknown Transformers loading class {loading_class}") from exc
            if loading_class == "Wav2Vec2ForCTC":
                config = transformers.AutoConfig.from_pretrained(
                    spec.checkpoint,
                    revision=spec.revision,
                    local_files_only=local_files_only,
                )
                # SpecAugment is inactive in eval mode. Disabling it before model
                # construction also avoids creating a random masked-spec vector
                # that fine-tuned CTC checkpoints intentionally do not contain.
                config.mask_time_prob = 0.0
                config.mask_feature_prob = 0.0
                model, loading_info = model_class.from_pretrained(
                    spec.checkpoint,
                    revision=spec.revision,
                    local_files_only=local_files_only,
                    config=config,
                    output_loading_info=True,
                )
                self.loading_info = {
                    key: sorted(value) if isinstance(value, set) else value
                    for key, value in loading_info.items()
                }
            else:
                model = model_class.from_pretrained(
                    spec.checkpoint,
                    revision=spec.revision,
                    local_files_only=local_files_only,
                )
        self.processor = processor
        self.model = model.to(device=device, dtype=self.torch_dtype).eval()
        # Keep the checkpoint's native task wrapper for faithful loading and
        # provenance, but run extraction against its underlying speech encoder.
        # For bare encoder classes, ``base_model`` is the model itself.
        self.encoder = self.model.base_model

    def frame_timing_samples(self) -> tuple[int, float]:
        """Return convolutional frame stride and receptive-field center in samples."""
        kernels = tuple(self.encoder.config.conv_kernel)
        strides = tuple(self.encoder.config.conv_stride)
        if len(kernels) != len(strides):
            raise RuntimeError("HuBERT convolution kernel and stride lengths differ")
        jump, receptive_field = 1, 1
        for kernel, stride in zip(kernels, strides):
            receptive_field += (int(kernel) - 1) * jump
            jump *= int(stride)
        return jump, (receptive_field - 1) / 2

    def prepare_audio(self, audio: np.ndarray) -> tuple[np.ndarray, str]:
        """Apply feature-extractor normalization once before audio is chunked."""
        audio = np.asarray(audio, dtype=np.float32)
        if getattr(self.processor, "do_normalize", False):
            variance = np.var(audio)
            audio = (audio - np.mean(audio)) / np.sqrt(variance + 1e-7)
            return audio.astype(np.float32, copy=False), "global_zero_mean_unit_variance"
        return audio, "none"

    def extract(self, audio):
        inputs = self.processor(
            audio,
            sampling_rate=self.spec.sample_rate_hz,
            return_tensors="pt",
        )
        return self._forward(inputs.input_values)

    def extract_prepared(self, audio):
        """Extract already globally normalized audio without per-chunk renormalization."""
        import torch

        values = torch.as_tensor(np.asarray(audio), dtype=torch.float32).unsqueeze(0)
        return self._forward(values)

    def _forward(self, input_values):
        import torch

        autocast = (
            torch.autocast(device_type="cuda", dtype=self.torch_dtype)
            if self.device.startswith("cuda") and self.dtype == "float16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = self.encoder(
                input_values=input_values.to(self.device),
                output_hidden_states=True,
                return_dict=True,
            )
        states = output.hidden_states
        if not states:
            raise RuntimeError("Model returned no hidden states")
        arrays = [
            state[0].detach().cpu().numpy().astype(self.numpy_dtype, copy=False)
            for state in states
        ]
        dimensions = {array.shape[1] for array in arrays}
        lengths = {array.shape[0] for array in arrays}
        if len(dimensions) != 1 or len(lengths) != 1:
            raise RuntimeError("Hidden-state shapes are inconsistent across layers")
        expected_count = getattr(self.encoder.config, "num_hidden_layers", None)
        if expected_count is not None and len(arrays) != expected_count + 1:
            raise RuntimeError(
                f"Expected {expected_count + 1} representations from runtime config, "
                f"received {len(arrays)}"
            )
        expected_width = getattr(self.encoder.config, "hidden_size", None)
        if expected_width is not None and next(iter(dimensions)) != expected_width:
            raise RuntimeError(
                f"Expected hidden width {expected_width}, received {next(iter(dimensions))}"
            )
        names = ["layer_00_input"] + [
            f"layer_{index:02d}_transformer" for index in range(1, len(arrays))
        ]
        config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else {}
        return dict(zip(names, arrays)), {
            "checkpoint": self.spec.checkpoint,
            "requested_revision": self.spec.revision,
            "resolved_commit_hash": getattr(self.model.config, "_commit_hash", None),
            "representation_count": len(arrays),
            "hidden_dimension": next(iter(dimensions)),
            "frame_count": next(iter(lengths)),
            "inference_dtype": self.dtype,
            "model_config": config,
        }

