

This repository contains the source code and stimulus data for an ICASSP 2027 submission in **Benchmark and evaluation of spoken language**. The benchmark measures how acoustics, prosody, phonetics, word timing, and recording onset contribute to the held-out prediction of layer activations. Conditional contribution is the difference in held-out R² between the full model and a fit omitting one predictor family. Word timing consists of boundaries and durations.

## Repository layout

- `scripts/` contains the command-line entry points listed below. Run them from the repository root.
- `src/speech_strf/` contains the importable implementation shared by those commands and the tests. Internal names containing `stage4` date from manuscript revision. They implement the submitted grouped analyses, controls, and result export. The public commands below use descriptive names.
- `stimuli/` contains the 12 paired Mandarin WAV and Praat TextGrid files used by the benchmark. The TextGrids provide the phone and word intervals. Check any applicable source permissions before redistributing the recordings.
- `configs/` defines the input files and benchmark settings. `tests/` checks preprocessing, model adapters, fitting, and inference on small inputs.

## 1. Set up a local environment

Use Python 3.10–3.13 on macOS or Linux. Windows users can run the same commands under WSL. Allow space for nine model downloads, activation stores, and fit outputs. Full fitting is substantially slower than the synthetic check.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,test]'
python scripts/validate_model_registry.py
python scripts/run_paper_analysis.py synthetic-test
```

The registry defaults to CPU and float32. You can choose a supported device and dtype in `configs/models.yaml` before extracting. Keep the same settings during the fit. The first extraction needs internet access for checkpoint downloads. To use a pre-downloaded complete checkpoint snapshot, pass `--checkpoint /path/to/snapshot` to both the extraction and fit commands for that model.

## 2. Validate inputs and extract features

```bash
python scripts/build_manifest.py --config configs/data.yaml
python scripts/extract_features.py --config configs/features.yaml
python scripts/extract_rich_features.py --config configs/features_rich.yaml
```

The manifest groups all frames by recording. Inspect `outputs/validation_report.json` if validation fails. The base features and activations share a 50 Hz time grid. The original predictor design has 144 columns before lag expansion, comprising 41 acoustic, 3 prosodic, 96 phonetic, 3 word-timing, and 1 recording-onset column. The rich feature archive adds spectral dynamics and descriptors for the richer-acoustics control.

## 3. Extract all nine checkpoints and fit the grouped reference models

These are the registry keys and model IDs. HuBERT Large writes to `outputs/hubert_large_refactor_rerun/`.

- `hubert_base` — `facebook/hubert-base-ls960`
- `hubert_large_reference` — `facebook/hubert-large-ls960-ft`
- `wav2vec2_base` — `facebook/wav2vec2-base-960h`
- `wav2vec2_large` — `facebook/wav2vec2-large-960h-lv60-self`
- `wavlm_base_plus` — `microsoft/wavlm-base-plus`
- `wavlm_large` — `microsoft/wavlm-large`
- `data2vec_audio_base` — `facebook/data2vec-audio-base-960h`
- `xls_r_300m` — `facebook/wav2vec2-xls-r-300m`
- `whisper_medium_encoder` — `openai/whisper-medium` (encoder only)

```bash
set -e
for model in hubert_base hubert_large_reference wav2vec2_base wav2vec2_large \
  wavlm_base_plus wavlm_large data2vec_audio_base xls_r_300m whisper_medium_encoder
do
  python scripts/run_model_pipeline.py --model "$model" --stage extract --allow-download
  python scripts/run_model_pipeline.py --model "$model" --stage fit
done
```

Extraction stores canonical layer activations and model metadata. The grouped reference fits provide fold-level sensitivity results and the original fold-specific ridge penalties for the fixed-alpha comparison. Checkpoint revisions are recorded in the extraction metadata where available. The supplied registry points to the public `main` revisions. The original environment and checkpoint hashes are unavailable in this repository, so exact numerical agreement also depends on the downloaded checkpoint and runtime.

## 4. Run the paper analyses

```bash
python scripts/check_model_provenance.py
python scripts/run_paper_analysis.py audit
python scripts/run_paper_analysis.py fit hubert_base original
python scripts/resolve_paper_layers.py
python scripts/run_paper_controls.py observed
python scripts/run_paper_controls.py null
```

The HuBERT Base analysis covers its input and all 12 transformer layers and includes leave-one-recording-out (LORO) sensitivity. The observed command prepares fixed-alpha refits and runs the richer-acoustics and equal-rank controls for HuBERT Base, wav2vec 2.0 Large, WavLM Large, data2vec Audio Base, XLS-R 300M, and Whisper Medium, plus zero-lag fits for all nine checkpoints. Each of these controls samples input, middle, and final depth. The null command runs 20 within-recording circular shifts at middle depth for HuBERT Base, wav2vec 2.0 Large, WavLM Large, and Whisper Medium, reusing the observed penalties.

The five-lag design uses −200, −100, 0, 100 and 200 ms. Fits use the configured grouped folds, training-only fitting of preprocessing, at most 30 target principal components, and the four candidate ridge penalties. Completed analysis units are reused after integrity checks. The CLI supports individual fits for troubleshooting. For example, `python scripts/run_paper_analysis.py fit hubert_base original --layer layer_00_input` fits only the input representation after the nine-model audit can pass.

## 5. Export and check the results

```bash
python scripts/export_paper_results.py
python scripts/export_paper_results.py --verify
```

The default export is under `outputs/stage4_revision/handoff/`. See `tables/` for the recording-level, layer-level, split-sensitivity, and PCA-coverage outputs, and `verification/missing_incomplete_units.csv` for gaps. Use `--archive-existing` when exporting again to the same location. The export writes recording-aggregated exact sign-flip tests, bootstrap intervals, and Benjamini–Hochberg adjusted values to `outputs/stage4_revision/paper_summary/aggregate_inference.csv`. `checkpoint_omissions_by_fold.csv` reports the checkpoint-omission sensitivity of the grouped reference fits.

For Figure 1, use the model registry, extraction metadata, and the predictor dimensions above. For the numerical inputs to Figure 2, use the HuBERT Base summary (A), richer-acoustics, equal-rank, and zero-lag recording tables (B–D), the paired observed-minus-shifted table (E), and `pca_coverage_summary.csv` (F). The publication figure layout and original fit archives are not bundled. As a check against the submission, the all-layer HuBERT Base means round to approximately 0.304 for acoustics, 0.0237 for prosody, 0.0200 for phonetics, 0.00281 for word timing, and 0.000240 for recording onset.

## Configuration reference

- `configs/data.yaml` selects the recordings, their annotations, and the manifest paths.
- `configs/features.yaml` defines the original feature archives.
- `configs/features_rich.yaml` defines the richer-acoustics control archives.
- `configs/models.yaml` selects the nine checkpoints and the extraction runtime.
- `configs/analysis.yaml` defines the grouped reference fits and supplies the contract checked before reusing their ridge penalties.
- `configs/paper_analysis.yaml` defines the recording-level fits, control variants, and inference options used by the paper workflow.

Run `python -m pytest` after installing the test and model extras if you want to verify the implementation on small fixtures.

