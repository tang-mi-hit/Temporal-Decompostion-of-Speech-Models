from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .adapters import StimulusRecord, build_adapter, configured_extraction_signature
from .alignments import parse_textgrid
from .audio import inspect_audio, load_standardized
from .comparability import build_comparability_contract, write_json
from .design_matrix import lagged_design
from .figures import make_result_figures
from .fit_encoding import nested_group_encoding
from .model_registry import RegistryEntry
from .provenance import load_config, sha256_file, write_model_run_metadata


def model_output_dir(entry: RegistryEntry) -> Path:
    return Path(entry.values.get("output_dir", f"outputs/{entry.key}"))


def output_identity_mismatches(
    entry: RegistryEntry,
    prior_metadata: dict,
    feature_config_path: str | Path,
    analysis_config_path: str | Path,
) -> dict[str, tuple[object, object]]:
    """Compare an existing run with the requested model and analysis identity."""
    observed_model = prior_metadata.get("model", {})
    observed_key = observed_model.get("key", observed_model.get("model_key"))
    expected_model = {
        "model_key": (observed_key, entry.key),
        "model_id": (observed_model.get("model_id"), entry.model_id),
        "revision": (observed_model.get("revision"), entry.revision),
    }
    mismatches = {
        key: values
        for key, values in expected_model.items()
        if values[0] != values[1]
    }
    expected_hashes = {
        "feature_config_sha256": sha256_file(feature_config_path),
        "analysis_config_sha256": sha256_file(analysis_config_path),
    }
    mismatches.update(
        {
            key: (prior_metadata.get(key), value)
            for key, value in expected_hashes.items()
            if prior_metadata.get(key) != value
        }
    )
    return mismatches


def extract_model(
    entry: RegistryEntry,
    manifest_path: str | Path,
    validation_report_path: str | Path,
    *,
    recording_ids: list[str] | None = None,
    allow_download: bool = False,
    overwrite: bool = False,
    output_dir: str | Path | None = None,
) -> Path:
    output = Path(output_dir) if output_dir else model_output_dir(entry)
    locked_destinations = {
        Path("outputs").resolve(),
        model_output_dir(entry).resolve(),
    }
    if entry.values.get("locked_reference") and output.resolve() in locked_destinations:
        raise RuntimeError(
            f"{entry.key} is a locked reference; provide a new --output directory "
            "for a regression rerun"
        )
    validation = json.loads(Path(validation_report_path).read_text())
    if not validation["valid"]:
        raise RuntimeError("Input validation is not valid; model extraction refused")
    manifest = pd.read_csv(manifest_path)
    if recording_ids:
        missing = set(recording_ids) - set(manifest["recording_id"])
        if missing:
            raise ValueError(f"Unknown recording IDs: {sorted(missing)}")
        manifest = manifest[manifest["recording_id"].isin(recording_ids)]
    adapter = build_adapter(entry)
    adapter.load_model(local_files_only=not allow_download)
    resolved_revision = getattr(getattr(adapter.model, "config", None), "_commit_hash", None)
    expected_signature = adapter.extraction_signature(resolved_revision)
    store_path = output / "activations.h5"
    rows = []
    first_metadata = None
    for row in manifest.to_dict("records"):
        duration = inspect_audio(row["audio_path"])["duration_seconds"]
        if store_path.exists() and not overwrite:
            with h5py.File(store_path) as store:
                if row["recording_id"] in store and store[
                    row["recording_id"]
                ].attrs.get("complete", False):
                    group = store[row["recording_id"]]
                    cached = json.loads(group.attrs["metadata_json"])
                    cached_signature = cached.get("extraction_signature")
                    if cached_signature != expected_signature:
                        raise RuntimeError(
                            f"Incompatible cached {row['recording_id']} extraction "
                            "signature; "
                            "use --overwrite to recompute"
                        )
                    first_metadata = first_metadata or cached
                    rows.append(
                        {
                            "recording_id": row["recording_id"],
                            "model_key": entry.key,
                            "status": "skipped_complete",
                            "duration_seconds": duration,
                            "native_frames": len(group["native_timestamps"]),
                            "canonical_frames": len(group["canonical_timestamps"]),
                            "layer_count": len(
                                json.loads(group.attrs["layer_names_json"])
                            ),
                            "audio_metadata_json": None,
                            "metadata_json": json.dumps(cached),
                        }
                    )
                    continue
        if entry.input_modality == "audio":
            audio, audio_metadata = load_standardized(
                row["audio_path"], int(entry.sample_rate_hz)
            )
            intervals = None
        else:
            audio, audio_metadata = None, None
            intervals = parse_textgrid(row["alignment_path"])
        record = StimulusRecord(
            recording_id=row["recording_id"],
            duration_seconds=duration,
            audio=audio,
            intervals=intervals,
        )
        result = adapter.extract_hidden_states(record)
        status = adapter.save_activation_artifacts(store_path, record, overwrite)
        first_metadata = first_metadata or result.metadata
        rows.append(
            {
                "recording_id": row["recording_id"],
                "model_key": entry.key,
                "status": status,
                "duration_seconds": duration,
                "native_frames": len(result.native_times),
                "canonical_frames": len(result.canonical_times),
                "layer_count": len(result.native_states),
                "audio_metadata_json": (
                    json.dumps(audio_metadata) if audio_metadata is not None else None
                ),
                "metadata_json": json.dumps(result.metadata),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    extraction_manifest = output / "extraction_manifest.csv"
    current = pd.DataFrame(rows)
    if extraction_manifest.exists():
        current = pd.concat(
            [pd.read_csv(extraction_manifest), current], ignore_index=True
        ).drop_duplicates("recording_id", keep="last")
    current.sort_values("recording_id").to_csv(extraction_manifest, index=False)
    if first_metadata:
        write_json(
            {
                "model": entry.as_dict(),
                "layers": first_metadata["layers"],
                "observed_model_metadata": first_metadata,
            },
            output / "layer_metadata.json",
        )
    return store_path


def fit_model(
    entry: RegistryEntry,
    activation_store: str | Path,
    features_dir: str | Path,
    manifest_path: str | Path,
    analysis_config_path: str | Path,
    feature_config_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, dict]:
    output = Path(output_dir) if output_dir else model_output_dir(entry)
    config = load_config(analysis_config_path)
    feature_config = load_config(feature_config_path)
    contextual = (
        feature_config.get("features", {})
        .get("optional", {})
        .get("contextual_text_embeddings", {})
    )
    if entry.input_modality == "text" and contextual.get("enabled", False):
        raise ValueError(
            "Contextual text embeddings cannot be predictors for the BERT target baseline"
        )
    manifest = pd.read_csv(manifest_path)
    recording_ids = sorted(manifest["recording_id"].astype(str))
    xs, groups, families = [], [], None
    feature_times = {}
    expected_signature = configured_extraction_signature(entry)
    for recording_id in recording_ids:
        feature = np.load(Path(features_dir) / f"{recording_id}.npz")
        current_families = feature["families"].tolist()
        if families is not None and current_families != families:
            raise ValueError(f"Feature schema differs for {recording_id}")
        families = current_families
        xs.append(feature["matrix"])
        feature_times[recording_id] = feature["times"]
        groups.extend([recording_id] * len(feature["times"]))
    design, lagged_families = lagged_design(
        np.vstack(xs),
        np.asarray(groups),
        families,
        config["analysis"]["lags_seconds"],
        config["analysis_rate_hz"],
    )
    results = []
    all_kernels = {}
    with h5py.File(activation_store) as store:
        if set(recording_ids) - set(store):
            raise ValueError(
                f"Activation store is missing {sorted(set(recording_ids) - set(store))}"
            )
        first = store[recording_ids[0]]
        layers = json.loads(first.attrs["layer_names_json"])
        for recording_id in recording_ids:
            group = store[recording_id]
            stored_metadata = json.loads(group.attrs["metadata_json"])
            stored_signature = stored_metadata.get("extraction_signature", {})
            mismatches = {
                key: (stored_signature.get(key), value)
                for key, value in expected_signature.items()
                if key != "resolved_revision" and stored_signature.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"{recording_id} activation identity differs from registry: "
                    f"{mismatches}"
                )
            times = group["canonical_timestamps"][:]
            if times.shape != feature_times[recording_id].shape or not np.allclose(
                times, feature_times[recording_id], atol=1e-8, rtol=0
            ):
                raise ValueError(
                    f"Canonical activation/feature grids differ for {recording_id}"
                )
            if json.loads(group.attrs["layer_names_json"]) != layers:
                raise ValueError(f"Layer schema differs for {recording_id}")
        for layer in layers:
            targets = np.vstack(
                [store[recording_id]["canonical"][layer][:] for recording_id in recording_ids]
            )
            frame, kernels = nested_group_encoding(
                design,
                targets,
                np.asarray(groups),
                lagged_families,
                config,
                layer,
            )
            layer_dir = output / "fit" / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(layer_dir / "results.csv", index=False)
            np.savez_compressed(layer_dir / "kernels.npz", **kernels)
            results.append(frame)
            all_kernels[layer] = kernels
    combined = pd.concat(results, ignore_index=True)
    fit_dir = output / "fit"
    combined_path = fit_dir / "all_layers_results.csv"
    combined.to_csv(combined_path, index=False)
    contract = build_comparability_contract(
        manifest_path, features_dir, feature_config_path, analysis_config_path
    )
    write_json(contract, output / "comparability_contract.json")
    return combined_path, all_kernels


def figure_model(
    entry: RegistryEntry,
    results_path: str | Path,
    kernels: dict,
    output_dir: str | Path | None = None,
) -> Path:
    output = Path(output_dir) if output_dir else model_output_dir(entry)
    first_layer = sorted(kernels)[0]
    label = entry.key
    if entry.input_modality == "text":
        label += " (text-only baseline)"
    make_result_figures(
        pd.read_csv(results_path),
        kernels[first_layer],
        output / "figures",
        title_prefix=label,
    )
    return output / "figures"


def finalize_metadata(
    entry: RegistryEntry,
    output_dir: str | Path,
    manifest_path: str | Path,
    model_config_path: str | Path,
    feature_config_path: str | Path,
    analysis_config_path: str | Path,
    random_seed: int,
) -> Path:
    contract_path = Path(output_dir) / "comparability_contract.json"
    split_hash = None
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        split_hash = hashlib.sha256(
            json.dumps(
                contract["outer_test_groups"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    layer_metadata_path = Path(output_dir) / "layer_metadata.json"
    extra = {}
    if layer_metadata_path.exists():
        observed = json.loads(layer_metadata_path.read_text()).get(
            "observed_model_metadata", {}
        )
        extra["resolved_model_revision"] = observed.get("resolved_revision")
    return write_model_run_metadata(
        output_dir,
        entry.as_dict(),
        manifest_path=manifest_path,
        model_config_path=model_config_path,
        feature_config_path=feature_config_path,
        analysis_config_path=analysis_config_path,
        split_definition_hash=split_hash,
        random_seed=random_seed,
        extra=extra,
    )
