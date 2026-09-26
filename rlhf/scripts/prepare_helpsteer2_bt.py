#!/usr/bin/env python3
"""Prepare the controlled HelpSteer2 Bradley-Terry composition study.

The natural arm retains every non-tie preference pair.  Five controlled
arms keep the number of training pairs fixed while setting the fraction for
which the longer response is preferred to 0, 25, 50, 75, or 100 percent.
Validation and test data are never resampled; the existing prompt-disjoint
HelpSteer2 regression contract supplies the tokenizer lengths and the fixed
held-out response/pair artifacts used by the common evaluator.
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
import subprocess
from typing import Any, Mapping, Sequence

from fair_rlhf.data import stable_pair_id, stable_prompt_id
from fair_rlhf.embeddings import stable_response_id
from fair_rlhf.phase2_bt import (
    TOKENIZER_NATIVE_SERIALIZATION,
    configure_phase2_chat_template,
)


SCHEMA_VERSION = 1
DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
DEFAULT_MODEL_REVISION = "1605565b47bb9346c5515c34102e054115b4f98b"
RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/processed/phase1_pairs/pairs.jsonl"),
    )
    parser.add_argument(
        "--regression-root",
        type=Path,
        default=Path("data/processed/phase2_helpsteer2_regression"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/phase2_helpsteer2_bt_llama31"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")
    if args.output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_root} already exists; pass --overwrite to rebuild"
        )

    regression_manifest = _read_json(args.regression_root / "manifest.json")
    _validate_regression_contract(regression_manifest, args)
    response_rows = {
        row["response_id"]: row
        for split in ("train", "validation", "test")
        for row in _read_jsonl(args.regression_root / f"{split}.jsonl")
    }
    source_pairs = _read_jsonl(args.pairs)
    prepared = [
        row
        for index, pair in enumerate(source_pairs, start=1)
        if (row := _prepare_pair(pair, index, response_rows)) is not None
    ]
    by_split = {
        split: [row for row in prepared if row["source_split"] == split]
        for split in ("train", "validation")
    }
    if not by_split["train"] or not by_split["validation"]:
        raise ValueError("nonempty train and validation preferences are required")
    _validate_prompt_disjointness(by_split)

    tokenizer_audit = _tokenizer_audit(args, prepared)
    long_pool = [row for row in by_split["train"] if row["winner_length"] == "long"]
    short_pool = [row for row in by_split["train"] if row["winner_length"] == "short"]
    fixed_count = min(len(long_pool), len(short_pool))
    fixed_count -= fixed_count % 4
    if fixed_count <= 0:
        raise ValueError("the controlled sweep has no feasible fixed sample size")
    long_pool = _stable_order(long_pool, seed=args.seed, label="long")
    short_pool = _stable_order(short_pool, seed=args.seed, label="short")

    conditions: dict[str, list[dict[str, Any]]] = {"natural": by_split["train"]}
    for ratio in RATIOS:
        name = _condition_name(ratio)
        long_count = int(round(fixed_count * ratio))
        selected = long_pool[:long_count] + short_pool[: fixed_count - long_count]
        conditions[name] = _stable_order(selected, seed=args.seed, label=name)

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "status": "ready",
        "seed": args.seed,
        "fixed_controlled_train_pairs": fixed_count,
        "conditions": {},
    }
    for condition, train_rows in conditions.items():
        condition_root = args.output_root / condition
        condition_root.mkdir(parents=True)
        serializable_train = [_training_row(row) for row in train_rows]
        serializable_validation = [_training_row(row) for row in by_split["validation"]]
        artifacts = {
            "train": _write_jsonl(condition_root / "train.jsonl", serializable_train),
            "validation": _write_jsonl(
                condition_root / "validation.jsonl", serializable_validation
            ),
        }
        train_direction = Counter(row["winner_length"] for row in train_rows)
        condition_ratio = (
            None
            if condition == "natural"
            else train_direction["long"] / len(train_rows)
        )
        manifest = _manifest(
            args=args,
            regression_manifest=regression_manifest,
            tokenizer_audit=tokenizer_audit,
            condition=condition,
            condition_ratio=condition_ratio,
            fixed_count=fixed_count,
            train_rows=train_rows,
            validation_rows=by_split["validation"],
            artifacts=artifacts,
        )
        _write_json(condition_root / "manifest.json", manifest)
        summary["conditions"][condition] = {
            "root": str(condition_root),
            "train_pairs": len(train_rows),
            "long_winner_pairs": train_direction["long"],
            "short_winner_pairs": train_direction["short"],
            "equal_length_pairs": train_direction["equal"],
            "long_winner_ratio": condition_ratio,
            "manifest_sha256": _file_sha256(condition_root / "manifest.json"),
        }
    _write_json(args.output_root / "sweep_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate_regression_contract(
    manifest: Mapping[str, Any], args: argparse.Namespace
) -> None:
    if manifest.get("status") != "ready":
        raise ValueError("the regression data contract is not ready")
    if manifest["model"]["name"] != args.model:
        raise ValueError("model does not match the regression contract")
    if manifest["model"]["requested_revision"] != args.model_revision:
        raise ValueError("model revision does not match the regression contract")
    if int(manifest["sequence_policy"]["max_length"]) != args.max_length:
        raise ValueError("max-length does not match the regression contract")
    if int(manifest["split"]["prompt_leakage_count"]) != 0:
        raise ValueError("the regression contract reports prompt leakage")


def _prepare_pair(
    pair: Mapping[str, Any],
    source_row: int,
    responses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    split = pair.get("split")
    if split not in {"train", "validation"}:
        return None
    label = int(pair["preference_label"])
    if label == 0:
        return None
    prompt = _required_text(pair.get("prompt"), "prompt")
    prompt_id = stable_prompt_id(prompt)
    if pair.get("prompt_id") != prompt_id:
        raise ValueError(f"source row {source_row} has a mismatched prompt_id")
    left = _required_text(pair.get("response_left"), "response_left")
    right = _required_text(pair.get("response_right"), "response_right")
    left_id = stable_response_id(prompt, left)
    right_id = stable_response_id(prompt, right)
    if left_id not in responses or right_id not in responses:
        return None
    if label > 0:
        chosen, rejected = left, right
        chosen_id, rejected_id = left_id, right_id
        chosen_position = 1
        chosen_tokens, rejected_tokens = (
            int(pair["token_count_left"]),
            int(pair["token_count_right"]),
        )
    else:
        chosen, rejected = right, left
        chosen_id, rejected_id = right_id, left_id
        chosen_position = 2
        chosen_tokens, rejected_tokens = (
            int(pair["token_count_right"]),
            int(pair["token_count_left"]),
        )
    if chosen_tokens > rejected_tokens:
        winner_length = "long"
    elif chosen_tokens < rejected_tokens:
        winner_length = "short"
    else:
        winner_length = "equal"
    oriented_pair_id = stable_pair_id(prompt, chosen, rejected)
    sample_payload = f"helpsteer2_bt\0{split}\0{source_row}\0{oriented_pair_id}"
    return {
        "sample_id": sha256(sample_payload.encode()).hexdigest(),
        "source_split": split,
        "source_row": source_row,
        "prompt_id": prompt_id,
        "pair_id": oriented_pair_id,
        "context": [{"role": "user", "content": prompt}],
        "chosen": chosen,
        "rejected": rejected,
        "preference_strength": int(pair["preference_strength"]),
        "domain": "helpsteer2",
        "language": "en",
        "chosen_source_position": chosen_position,
        "chosen_sequence_length": int(responses[chosen_id]["sequence_length"]),
        "rejected_sequence_length": int(responses[rejected_id]["sequence_length"]),
        "chosen_response_token_count": chosen_tokens,
        "rejected_response_token_count": rejected_tokens,
        "winner_length": winner_length,
    }


def _training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "winner_length"}


def _tokenizer_audit(
    args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise SystemExit("Transformers is required to verify the tokenizer contract") from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        use_fast=True,
        local_files_only=True,
    )
    serialization = configure_phase2_chat_template(
        tokenizer,
        serialization_profile=TOKENIZER_NATIVE_SERIALIZATION,
    )
    # Verify a deterministic sample against the regression contract. Runtime
    # training verifies every sequence length again in its collator.
    sample = _stable_order(rows, seed=args.seed, label="tokenizer-audit")[:32]
    rendered: list[str] = []
    expected: list[int] = []
    for row in sample:
        for side in ("chosen", "rejected"):
            rendered.append(
                tokenizer.apply_chat_template(
                    [*row["context"], {"role": "assistant", "content": row[side]}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            expected.append(int(row[f"{side}_sequence_length"]))
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_length=True,
    )
    actual = [int(value) for value in encoded.get("length", [])]
    if not actual:
        actual = [len(ids) for ids in encoded["input_ids"]]
    if actual != expected:
        raise ValueError("tokenizer audit differs from the regression sequence lengths")
    return {
        "name": args.model,
        "requested_revision": args.model_revision,
        "resolved_revision": args.model_revision,
        "class": type(tokenizer).__name__,
        "chat_template_sha256": serialization["chat_template_sha256"],
        "serialization_profile": serialization["profile"],
        "serialization_source": serialization["source"],
    }


def _manifest(
    *,
    args: argparse.Namespace,
    regression_manifest: Mapping[str, Any],
    tokenizer_audit: Mapping[str, str],
    condition: str,
    condition_ratio: float | None,
    fixed_count: int,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    train_counts = Counter(row["winner_length"] for row in train_rows)
    validation_counts = Counter(row["winner_length"] for row in validation_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "status": "ready",
        "study": {
            "name": "helpsteer2_bt_comparative_separation",
            "condition": condition,
            "objective": "binary_bradley_terry",
            "long_winner_ratio": condition_ratio,
        },
        "source": {
            "dataset": "nvidia/HelpSteer2",
            "pairs": str(args.pairs),
            "pair_sha256": _file_sha256(args.pairs),
            "regression_contract": str(args.regression_root),
            "regression_manifest_sha256": _file_sha256(
                args.regression_root / "manifest.json"
            ),
            "domains": ["helpsteer2"],
        },
        "tokenizer": dict(tokenizer_audit),
        "sequence_policy": {
            "max_length": args.max_length,
            "overlength_pair_policy": "omit_if_either_response_was_omitted",
            "serialization": "user_assistant_chat",
        },
        "split_policy": {
            "unit": "prompt_id",
            "name": "reuse_phase1_prompt_split",
            "seed": args.seed,
            "post_length_prompt_leakage_count": 0,
            "paper_alignment_note": (
                "The existing prompt-disjoint HelpSteer2 split is reused; "
                "only training-pair composition changes across conditions."
            ),
        },
        "sampling": {
            "condition": condition,
            "seed": args.seed,
            "without_replacement": True,
            "fixed_controlled_train_pairs": fixed_count,
            "requested_long_winner_ratio": condition_ratio,
            "realized_long_winner_ratio": condition_ratio,
            "natural_uses_all_non_tie_pairs": condition == "natural",
            "equal_length_policy": (
                "retain" if condition == "natural" else "exclude_as_undefined"
            ),
        },
        "normalization_audit": {
            "excluded_tie_rows": (
                int(regression_manifest["source"]["pairs"])
                - len(train_rows)
                - len(validation_rows)
                if condition == "natural"
                else None
            ),
            "preference_strength_counts": dict(
                sorted(Counter(str(row["preference_strength"]) for row in train_rows).items())
            ),
            "chosen_source_position_counts": dict(
                sorted(Counter(str(row["chosen_source_position"]) for row in train_rows).items())
            ),
            "train_winner_length_counts": dict(sorted(train_counts.items())),
            "validation_winner_length_counts": dict(sorted(validation_counts.items())),
        },
        "length_audit": {
            "train": {"pairs_after_length_filter": len(train_rows)},
            "validation": {"pairs_after_length_filter": len(validation_rows)},
        },
        "artifacts": dict(artifacts),
        "heldout_evaluation": {
            "data_root": str(args.regression_root),
            "test_sha256": regression_manifest["artifacts"]["test"]["sha256"],
            "test_pairs_sha256": regression_manifest["artifacts"]["test_pairs"]["sha256"],
        },
        "software": {
            "python": platform.python_version(),
            "git_commit": _git_commit(),
        },
    }


def _validate_prompt_disjointness(
    rows: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    train_prompts = {row["prompt_id"] for row in rows["train"]}
    validation_prompts = {row["prompt_id"] for row in rows["validation"]}
    overlap = train_prompts & validation_prompts
    if overlap:
        raise ValueError(f"train/validation prompt leakage: {len(overlap)}")


def _stable_order(
    rows: Sequence[Mapping[str, Any]], *, seed: int, label: str
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: sha256(
            f"{seed}\0{label}\0{row['sample_id']}".encode()
        ).hexdigest(),
    )


def _condition_name(ratio: float) -> str:
    return f"long_{int(round(100 * ratio)):03d}"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digest = sha256()
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            stream.write(line)
            digest.update(line.encode())
    return {"path": str(path), "rows": len(rows), "sha256": digest.hexdigest()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
