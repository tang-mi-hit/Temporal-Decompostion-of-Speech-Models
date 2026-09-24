# HuBERT-first speech STRF pipeline

This phase provides a leakage-safe, configuration-driven pipeline for asking how
acoustic, prosodic, phonetic, onset, and word-level predictors explain held-out
HuBERT activations. It is methodologically inspired by Zhang et al., *Neuron*
(2026), DOI `10.1016/j.neuron.2025.10.011`; it is an independent implementation,
not an exact replication.

## Setup and smoke test

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[test]'
pytest
python3 scripts/smoke_test.py --config configs/smoke_test.yaml
```

The smoke command uses generated audio-like fixtures and makes a nested
group-CV results table plus SVG/PDF diagnostics without downloading a model.

## Real-data sequence

```bash
python3 scripts/build_manifest.py --config configs/data.yaml
python3 scripts/extract_features.py --config configs/features.yaml
# The completed legacy HuBERT extraction/fit/figures are now a locked reference.
python3 scripts/validate_model_registry.py
```

Extraction refuses to run unless the validation report is valid. Models are
local-cache-only unless `--allow-download` is explicit. Audio remains read-only;
do not commit or upload audio, annotations, transcripts, or activations.

The default 4090D extraction configuration uses CUDA FP16 with 20-second core
windows and one second of context on both sides. Activation frames are stitched
by their convolutional receptive-field centers; overlap frames are retained
exactly once. Completed recording groups are skipped on rerun. Use
`--overwrite` only to intentionally recompute them.

## Multi-model benchmark

`configs/models.yaml` is the model registry. Every nonreference model writes to
`outputs/<model-key>/`; the existing legacy `outputs/activations.h5`, fit
results, and figures are never modified. All adapters preserve native
timestamps and linearly interpolate audio-model states onto the same 50 Hz
audio-relative grid used by HuBERT. Downstream fitting uses only canonical-grid
states and the existing feature files, lags, grouped folds, metrics, reduced
models, and figure functions.

Validate registry structure and installed Transformers classes:

```bash
python3 scripts/validate_model_registry.py
```

A complete `hf download MODEL_ID --local-dir DIRECTORY` snapshot already
contains model weights, `config.json`, and the processor/tokenizer metadata.
No separate calculation creates these files. Transfer the entire directory;
manually copying only `pytorch_model.bin` or `model.safetensors` is insufficient.
BERT needs its tokenizer files and Whisper needs `preprocessor_config.json`.

Run one model after staging its checkpoint locally. Pass the same local
checkpoint identity to every stage:

```bash
MODEL=wavlm_base_plus
CHECKPOINT=/share/home/mitan/models/wavlm-base-plus
python3 scripts/run_model_pipeline.py --model "$MODEL" --checkpoint "$CHECKPOINT" --stage extract
python3 scripts/run_model_pipeline.py --model "$MODEL" --checkpoint "$CHECKPOINT" --stage fit
python3 scripts/run_model_pipeline.py --model "$MODEL" --checkpoint "$CHECKPOINT" --stage figures
python3 scripts/compare_to_hubert_reference.py --model wavlm_base_plus
```

The same command pattern applies to these enabled keys:

```text
hubert_base
wav2vec2_base
wav2vec2_large
wavlm_base_plus
wavlm_large
data2vec_audio_base
xls_r_300m
whisper_medium_encoder
bert_base_uncased
```

For example:

```bash
python3 scripts/run_model_pipeline.py --model bert_base_uncased --stage all
python3 scripts/compare_to_hubert_reference.py --model bert_base_uncased
```

`mms_1b_all` is registered but disabled pending license review. The resolved
public W2V-BERT identifier is `facebook/w2v-bert-2.0`, pinned in the registry,
but `w2v_bert_2` remains disabled until its native frontend timing is validated
with the actual checkpoint.

To add a model, add one registry entry. Reuse `generic_speech` for waveform
encoders with a Hugging Face hidden-state model and convolutional timing. Add a
new adapter only for a genuinely different input/timing contract.

### BERT baseline

BERT uses labeled forced-aligned word intervals. A fast tokenizer retains
word-to-wordpiece ownership; wordpieces are mean-pooled per layer. Context is
split at punctuation or a configured 0.5-second gap. Each word vector is held
constant over its `[start, end)` interval on the canonical grid, with zeros
outside aligned words. Coverage and unknown-token fraction are recorded.
BERT-derived values are targets only and are explicitly forbidden from the
predictor feature set. Figures label BERT as a text-only baseline. Because the
checkpoint is English uncased while the stimuli are Mandarin, its tokenizer
coverage requires careful interpretation.

### Comparability and HuBERT regression

Every completed fit writes `comparability_contract.json` and
`run_metadata.json`. Comparison checks stimulus IDs, canonical grid, feature
order/configuration, lag grid, held-out groups, reduced models, metrics, and
result schema. `comparability_report.json` is machine-readable.

Freeze a contract beside the completed legacy HuBERT outputs once:

```bash
python3 scripts/freeze_hubert_reference.py
```

For a non-destructive HuBERT refactor rerun, provide a separate output:

```bash
export HUBERT=/share/home/mitan/models/hubert-large-ls960-ft/hubert-large-ls960-ft
python3 scripts/run_model_pipeline.py \
  --model hubert_large_reference \
  --checkpoint "$HUBERT" \
  --output outputs/hubert_large_refactor_rerun \
  --stage extract
python3 scripts/run_model_pipeline.py \
  --model hubert_large_reference \
  --checkpoint "$HUBERT" \
  --output outputs/hubert_large_refactor_rerun \
  --stage fit
python3 scripts/run_model_pipeline.py \
  --model hubert_large_reference \
  --checkpoint "$HUBERT" \
  --output outputs/hubert_large_refactor_rerun \
  --stage figures
python3 scripts/compare_to_hubert_reference.py \
  --model hubert_large_reference \
  --candidate-root outputs/hubert_large_refactor_rerun \
  --run-hubert-regression
```

`--stage fit` consumes `activations.h5` produced by `--stage extract` in the
same output directory; it does not load or extract the checkpoint itself. In a
Slurm script using `set -u`, define or export `HUBERT` before the first
`"$HUBERT"` expansion.

Regression tolerances are `2e-3` absolute for FP16 activations, `1e-6` seconds
for timestamps, and `1e-4` for downstream summaries.

### Cross-model predictability summary

After each candidate has passed `compare_to_hubert_reference.py`, aggregate any
completed set without rerunning a model:

```bash
python3 scripts/summarize_model_comparison.py \
  --model hubert_base \
  --model wavlm_base_plus \
  --model whisper_medium_encoder
```

The command writes `outputs/model_comparison/model_comparison_summary.csv`,
layer and feature-family detail tables, paired fold differences from HuBERT
Large, SVG/PDF predictability figures, and source hashes. It refuses candidates
whose comparability report did not pass. “Higher predictability” means the fixed
feature set explains more variance in that representation; it is not a general
model-quality or causal claim.

Input validation treats a configured overhang of at most 30 ms as a warning only
when the interval label is empty. Original TextGrid times remain unchanged.
Labeled intervals, larger overruns, missing files, and malformed timing remain
errors that block extraction.

## Method choices

Paper-compatible choices explicitly represented in configuration are the 50 Hz
analysis grid, HuBERT layer 0 plus all returned transformer hidden states, and a
separate zero-lag analysis. Runtime inspection—not a hard-coded layer count—
records representation count, dimensions, observed frame rate, model config,
requested revision, and resolved commit hash.

Project extensions are configurable temporal lags, nested story-grouped CV,
training-fold-only normalization and PCA, per-unit or PCA targets, and
independently tuned full-versus-reduced ridge models. `ΔR²` is named
**conditional unique contribution** and is not interpreted causally.

Optional phoneme-sequence statistics, word frequency, lexical surprisal, and
contextual embeddings are disabled and unavailable until suitable Mandarin
transcripts/resources are supplied. Default temporal lags and ridge
hyperparameters are project choices, not claims about the paper.

The retained initial pool contains 12 recordings. `story17` and `story18` are
explicitly excluded in `configs/data.yaml` because they are nonsemantic control
conditions, not stimulus stories for this encoding analysis.

## Current scaling blockers

The configured `/share/workspace3/...` and `/share/home/mitan/...` mounts are
not visible in the Cloud Agent used to create this scaffold. Consequently,
`outputs/validation_report.json` records missing real inputs, and no real model
details are asserted. Before scaling, confirm TextGrid tier names and label
conventions, story grouping for segmented files, approved model-cache location,
compute/device policy, and the Mandarin lexical resources (if optional features
are desired).

