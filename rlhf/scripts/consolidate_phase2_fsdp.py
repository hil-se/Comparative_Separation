#!/usr/bin/env python3
"""Export a PyTorch 2.1 FSDP checkpoint as HF safetensor shards."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-numel", type=int, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--staging-root",
        type=Path,
        help=(
            "Optional node-local directory used to stage and validate the "
            "HF export before it is copied to output-dir."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == checkpoint:
        raise ValueError("output-dir must differ from the source checkpoint")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    sharded_state = checkpoint / "pytorch_model_fsdp_0"
    metadata_path = sharded_state / ".metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"FSDP metadata not found: {metadata_path}"
        )
    for required in ("config.json", "tokenizer_config.json"):
        if not (checkpoint / required).is_file():
            raise FileNotFoundError(
                f"checkpoint metadata file not found: {checkpoint / required}"
            )

    import torch

    target_dtype = getattr(torch, args.dtype)
    state_dict = _load_model_state_dict(
        sharded_state,
        target_dtype=target_dtype,
    )
    total_numel = sum(tensor.numel() for tensor in state_dict.values())
    if total_numel != args.expected_numel:
        raise ValueError(
            f"loaded checkpoint has {total_numel} elements, "
            f"expected {args.expected_numel}"
        )
    if not any(key.endswith("score.weight") for key in state_dict):
        raise ValueError("checkpoint is missing the scalar reward head")
    if not any(key.endswith("embed_tokens.weight") for key in state_dict):
        raise ValueError("checkpoint is missing token embeddings")

    floating_dtypes = {
        str(tensor.dtype)
        for tensor in state_dict.values()
        if tensor.is_floating_point()
    }
    if floating_dtypes != {str(target_dtype)}:
        raise ValueError(
            f"floating tensors have unexpected dtypes: {floating_dtypes}"
        )

    staging_name = (
        f".{output_dir.name}.tmp-"
        f"{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    external_staging = args.staging_root is not None
    if external_staging:
        staging_root = args.staging_root.resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = staging_root / staging_name
    else:
        staging_dir = output_dir.with_name(staging_name)
    if staging_dir.exists():
        raise FileExistsError(
            f"staging directory already exists: {staging_dir}"
        )
    staging_dir.mkdir(parents=True)
    _copy_checkpoint_metadata(checkpoint, staging_dir)
    weight_files = _save_safetensor_shards(
        state_dict,
        staging_dir,
        max_shard_size=args.max_shard_size,
    )
    _validate_safetensor_shards(
        state_dict,
        staging_dir,
        weight_files,
    )

    dtype_numel: Counter[str] = Counter()
    for tensor in state_dict.values():
        dtype_numel[str(tensor.dtype)] += tensor.numel()
    weight_artifacts = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in weight_files
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _now(),
        "status": "ready",
        "source_checkpoint": str(checkpoint),
        "source_sharded_state": str(sharded_state),
        "source_metadata_sha256": _file_sha256(metadata_path),
        "output_directory": str(output_dir),
        "staging_directory": str(staging_dir),
        "target_dtype": str(target_dtype),
        "tensor_count": len(state_dict),
        "total_numel": total_numel,
        "dtype_numel": dict(sorted(dtype_numel.items())),
        "max_shard_size": args.max_shard_size,
        "weight_artifacts": weight_artifacts,
        "weight_bytes": sum(
            int(artifact["bytes"]) for artifact in weight_artifacts
        ),
    }
    _write_json(staging_dir / "consolidation_manifest.json", manifest)
    del state_dict

    if external_staging:
        transfer_dir = output_dir.with_name(
            f".{output_dir.name}.transfer-"
            f"{os.environ.get('SLURM_JOB_ID', os.getpid())}"
        )
        if transfer_dir.exists():
            raise FileExistsError(
                f"transfer directory already exists: {transfer_dir}"
            )
        shutil.copytree(staging_dir, transfer_dir, copy_function=shutil.copy2)
        _validate_copied_export(transfer_dir, weight_artifacts)
        transfer_dir.replace(output_dir)
        shutil.rmtree(staging_dir)
    else:
        staging_dir.replace(output_dir)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _load_model_state_dict(
    sharded_state: Path,
    *,
    target_dtype: Any,
) -> dict[str, Any]:
    import torch
    import torch.distributed.checkpoint as distributed_checkpoint
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint._traverse import set_element
    from torch.distributed.checkpoint.default_planner import (
        DefaultLoadPlanner,
    )
    from torch.distributed.checkpoint.metadata import (
        TensorStorageMetadata,
    )

    class CastingEmptyStateDictLoadPlanner(DefaultLoadPlanner):
        def set_up_planner(
            self,
            state_dict: dict[str, Any],
            metadata: Any,
            is_coordinator: bool,
        ) -> None:
            if state_dict:
                raise ValueError("the destination state dict must be empty")
            planner_data = metadata.planner_data or {}
            for key, storage_metadata in (
                metadata.state_dict_metadata.items()
            ):
                value: Any = storage_metadata
                if isinstance(
                    storage_metadata,
                    TensorStorageMetadata,
                ):
                    source_dtype = storage_metadata.properties.dtype
                    empty_scalar = torch.empty((), dtype=source_dtype)
                    destination_dtype = (
                        target_dtype
                        if empty_scalar.is_floating_point()
                        else source_dtype
                    )
                    value = torch.empty(
                        storage_metadata.size,
                        dtype=destination_dtype,
                    )
                if key in planner_data:
                    set_element(state_dict, planner_data[key], value)
                else:
                    state_dict[key] = value
            super().set_up_planner(
                state_dict,
                metadata,
                is_coordinator,
            )

    loaded: dict[str, Any] = {}
    distributed_checkpoint.load_state_dict(
        state_dict=loaded,
        storage_reader=FileSystemReader(sharded_state),
        no_dist=True,
        planner=CastingEmptyStateDictLoadPlanner(),
    )
    if set(loaded) != {"model"} or not isinstance(loaded["model"], dict):
        raise ValueError(
            "expected the DCP checkpoint to contain one nested model state"
        )
    model_state = loaded["model"]
    invalid = {
        key: type(value).__name__
        for key, value in model_state.items()
        if not isinstance(value, torch.Tensor)
    }
    if invalid:
        raise TypeError(
            "model state contains non-tensor values: "
            f"{dict(list(invalid.items())[:5])}"
        )
    return model_state


def _copy_checkpoint_metadata(source: Path, destination: Path) -> None:
    excluded_prefixes = (
        "model",
        "pytorch_model",
        "sharded_checkpoint_report",
    )
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(excluded_prefixes):
            continue
        shutil.copy2(path, destination / path.name)


def _save_safetensor_shards(
    state_dict: dict[str, Any],
    output_dir: Path,
    *,
    max_shard_size: str,
) -> list[Path]:
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    weight_files: list[Path] = []
    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {
            name: state_dict[name].contiguous()
            for name in tensor_names
        }
        path = output_dir / filename
        save_file(shard, path, metadata={"format": "pt"})
        weight_files.append(path)
    if split.is_sharded:
        index = {
            "metadata": split.metadata,
            "weight_map": split.tensor_to_filename,
        }
        _write_json(
            output_dir / "model.safetensors.index.json",
            index,
        )
    return sorted(weight_files)


def _validate_safetensor_shards(
    state_dict: dict[str, Any],
    output_dir: Path,
    weight_files: list[Path],
) -> None:
    from safetensors import safe_open

    observed: dict[str, tuple[tuple[int, ...], str]] = {}
    for path in weight_files:
        with safe_open(path, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key in observed:
                    raise ValueError(f"duplicate exported tensor: {key}")
                tensor_slice = stream.get_slice(key)
                observed[key] = (
                    tuple(tensor_slice.get_shape()),
                    tensor_slice.get_dtype(),
                )
    if set(observed) != set(state_dict):
        missing = set(state_dict) - set(observed)
        extra = set(observed) - set(state_dict)
        raise ValueError(
            f"exported tensor inventory differs: "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )
    for key, tensor in state_dict.items():
        observed_shape, _ = observed[key]
        if observed_shape != tuple(tensor.shape):
            raise ValueError(f"exported shape differs for {key}")
    if not (output_dir / "config.json").is_file():
        raise FileNotFoundError("export is missing config.json")
    if not (output_dir / "tokenizer_config.json").is_file():
        raise FileNotFoundError("export is missing tokenizer_config.json")


def _validate_copied_export(
    output_dir: Path,
    weight_artifacts: list[dict[str, Any]],
) -> None:
    """Verify that a cross-filesystem scratch copy is byte-identical."""

    for artifact in weight_artifacts:
        path = output_dir / str(artifact["path"])
        if not path.is_file():
            raise FileNotFoundError(f"copied export is missing {path.name}")
        if path.stat().st_size != int(artifact["bytes"]):
            raise ValueError(f"copied export size differs for {path.name}")
        if _file_sha256(path) != str(artifact["sha256"]):
            raise ValueError(f"copied export hash differs for {path.name}")
    for required in (
        "config.json",
        "tokenizer_config.json",
        "consolidation_manifest.json",
    ):
        if not (output_dir / required).is_file():
            raise FileNotFoundError(
                f"copied export is missing {required}"
            )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
