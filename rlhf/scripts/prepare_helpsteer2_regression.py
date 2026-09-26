#!/usr/bin/env python3
"""Prepare a fine-tuned HelpSteer2 pointwise-regression contract.

The input is the existing normalized, prompt-disjoint HelpSteer2 pair
artifact.  Responses are deduplicated, tokenized with the pinned Llama 3.1
tokenizer, and written as response-level regression records.  FairReweighing
weights are estimated from the training responses only using the
reference-compatible density-balance rule in ``fair_rlhf.cp``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import shutil
from typing import Any, Mapping

import numpy as np

from fair_rlhf.cp import DensityBalance
from fair_rlhf.embeddings import stable_response_id
from fair_rlhf.data import stable_prompt_id


SCHEMA_VERSION = 1
DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
DEFAULT_MODEL_REVISION = "1605565b47bb9346c5515c34102e054115b4f98b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare prompt-disjoint HelpSteer2 response-level regression "
            "data with train-only density FairReweighing weights."
        )
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/processed/phase1_pairs/pairs.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/phase2_helpsteer2_regression"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument(
        "--density-model",
        choices=("Neighbor", "Kernel"),
        default="Neighbor",
    )
    parser.add_argument("--density-radius", type=float, default=0.5)
    parser.add_argument("--density-bandwidth", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_length <= 0 or args.tokenizer_batch_size <= 0:
        raise ValueError("max-length and tokenizer-batch-size must be positive")
    if args.output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_root} already exists; pass --overwrite to rebuild"
        )

    pairs = _read_jsonl(args.pairs)
    responses, pair_index = _deduplicate_responses(pairs)
    _attach_token_lengths(
        responses,
        model=args.model,
        revision=args.model_revision,
        max_length=args.max_length,
        batch_size=args.tokenizer_batch_size,
    )
    retained_ids = {row["response_id"] for row in responses}
    for pair in pair_index:
        if pair["split"] == "test" and not {
            pair["left_response_id"],
            pair["right_response_id"],
        }.issubset(retained_ids):
            raise ValueError(
                "a test pair references a response omitted by the length policy"
            )

    train = [row for row in responses if row["split"] == "train"]
    validation = [row for row in responses if row["split"] == "validation"]
    test = [row for row in responses if row["split"] == "test"]
    if not train or not validation or not test:
        raise ValueError("train, validation, and test responses are required")
    _validate_prompt_splits(responses)

    train_weights = DensityBalance(
        model=args.density_model,
        radius=args.density_radius,
        bandwidth=args.density_bandwidth,
    ).weight(
        np.asarray([[row["response_token_count"]] for row in train]),
        np.asarray([row["helpfulness"] for row in train]),
    )
    for row, weight in zip(train, train_weights, strict=True):
        row["fair_reweighing_weight"] = float(weight)
    for row in (*validation, *test):
        row["fair_reweighing_weight"] = 1.0

    args.output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for split, rows in (("train", train), ("validation", validation), ("test", test)):
        artifacts[split] = _write_jsonl(args.output_root / f"{split}.jsonl", rows)
    test_pairs = [
        row
        for row in pair_index
        if row["split"] == "test"
        and row["left_response_id"] in retained_ids
        and row["right_response_id"] in retained_ids
    ]
    artifacts["test_pairs"] = _write_jsonl(
        args.output_root / "test_pairs.jsonl",
        test_pairs,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "study": {
            "name": "helpsteer2_finetuned_regression_fairreweighing",
            "task": "pointwise_helpfulness_regression",
            "split_unit": "prompt_id",
            "source_pairs": str(args.pairs),
        },
        "source": {
            "pair_sha256": _file_sha256(args.pairs),
            "pairs": len(pairs),
            "responses": len(responses),
            "response_rating_scale": [0, 4],
        },
        "model": {
            "name": args.model,
            "requested_revision": args.model_revision,
        },
        "sequence_policy": {
            "max_length": args.max_length,
            "overlength_response_policy": "omit",
            "serialization": "user_assistant_chat",
        },
        "split": {
            "train_responses": len(train),
            "validation_responses": len(validation),
            "test_responses": len(test),
            "train_prompts": len({row["prompt_id"] for row in train}),
            "validation_prompts": len({row["prompt_id"] for row in validation}),
            "test_prompts": len({row["prompt_id"] for row in test}),
            "prompt_leakage_count": _prompt_leakage_count(responses),
        },
        "fair_reweighing": {
            "implementation": "fair_rlhf.cp.DensityBalance",
            "reference": (
                "https://github.com/hil-se/Comparative_Separation/blob/main/cp.py"
            ),
            "companion_reference": (
                "https://github.com/hil-se/Comparative_Separation/blob/main/"
                "real/src/density_balance.py"
            ),
            "treatment": "Reweighing",
            "sensitive_attribute": "response_token_count",
            "target": "helpfulness",
            "density_model": args.density_model,
            "radius": args.density_radius,
            "bandwidth": args.density_bandwidth,
            "fit_scope": "training_responses_only",
            "weight_minimum": float(train_weights.min()),
            "weight_mean": float(train_weights.mean()),
            "weight_maximum": float(train_weights.max()),
            "effective_sample_size": float(
                train_weights.sum() ** 2 / np.square(train_weights).sum()
            ),
        },
        "artifacts": artifacts,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": _git_commit(),
        },
    }
    _write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _deduplicate_responses(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    responses: dict[str, dict[str, Any]] = {}
    pair_index: list[dict[str, Any]] = []
    for row_number, pair in enumerate(pairs, start=1):
        prompt = _required_text(pair.get("prompt"), "prompt")
        prompt_id = pair.get("prompt_id")
        if prompt_id != stable_prompt_id(prompt):
            raise ValueError(f"pair {row_number} has a mismatched prompt_id")
        split = pair.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"pair {row_number} has an invalid split")
        left_id = _add_response(responses, pair, "left", prompt, prompt_id, split)
        right_id = _add_response(responses, pair, "right", prompt, prompt_id, split)
        pair_index.append(
            {
                "pair_id": pair.get("pair_id"),
                "prompt_id": prompt_id,
                "prompt": prompt,
                "split": split,
                "preference_label": int(pair["preference_label"]),
                "preference_strength": int(pair["preference_strength"]),
                "left_response_id": left_id,
                "right_response_id": right_id,
                "left_token_count": int(pair["token_count_left"]),
                "right_token_count": int(pair["token_count_right"]),
            }
        )
    ordered = sorted(responses.values(), key=lambda row: row["response_id"])
    return ordered, pair_index


def _add_response(
    responses: dict[str, dict[str, Any]],
    pair: Mapping[str, Any],
    side: str,
    prompt: str,
    prompt_id: str,
    split: str,
) -> str:
    response = _required_text(pair.get(f"response_{side}"), f"response_{side}")
    response_id = stable_response_id(prompt, response)
    candidate = {
        "response_id": response_id,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "response": response,
        "helpfulness": float(pair[f"helpfulness_{side}"]),
        "response_token_count": int(pair[f"token_count_{side}"]),
        "split": split,
    }
    existing = responses.get(response_id)
    if existing is not None:
        for key in ("prompt_id", "prompt", "helpfulness", "response_token_count", "split"):
            if existing[key] != candidate[key]:
                raise ValueError(
                    f"inconsistent metadata for response {response_id}: {key}"
                )
    else:
        responses[response_id] = candidate
    return response_id


def _attach_token_lengths(
    rows: list[dict[str, Any]],
    *,
    model: str,
    revision: str,
    max_length: int,
    batch_size: int,
) -> None:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Transformers is required to prepare the regression contract"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        revision=revision,
        use_fast=True,
    )
    retained: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for row in batch
        ]
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        for row, ids in zip(batch, encoded["input_ids"], strict=True):
            row["sequence_length"] = int(len(ids))
            if len(ids) <= max_length:
                retained.append(row)
    rows[:] = retained


def _validate_prompt_splits(rows: list[dict[str, Any]]) -> None:
    split_by_prompt: dict[str, str] = {}
    for row in rows:
        previous = split_by_prompt.setdefault(row["prompt_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("prompt appears in multiple experimental splits")


def _prompt_leakage_count(rows: list[dict[str, Any]]) -> int:
    split_prompts = {
        split: {row["prompt_id"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    return sum(
        len(split_prompts[left] & split_prompts[right])
        for index, left in enumerate(split_prompts)
        for right in list(split_prompts)[index + 1 :]
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    digest = sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return {"path": str(path), "rows": len(rows), "sha256": digest.hexdigest()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

