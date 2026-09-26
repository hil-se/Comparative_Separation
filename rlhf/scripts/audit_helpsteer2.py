#!/usr/bin/env python3
"""Download, normalize, and audit HelpSteer2 without committing dataset rows."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from typing import Any

from fair_rlhf.data import (
    NormalizedPair,
    assign_prompt_splits,
    attach_token_counts,
    audit_pairs,
    normalize_helpsteer2_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join HelpSteer2 preferences to their exact pointwise ratings, "
            "create prompt-disjoint experimental splits, and write an audit report."
        )
    )
    parser.add_argument("--dataset", default="nvidia/HelpSteer2")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision or commit hash.",
    )
    parser.add_argument(
        "--preference-data-dir",
        default="preference",
        help="Data directory containing the preference JSONL.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--tokenizer",
        help=(
            "Optional Hugging Face tokenizer. If omitted, alignment, position, "
            "and split checks run without token-length diagnostics."
        ),
    )
    parser.add_argument(
        "--tokenizer-revision",
        default="main",
        help="Tokenizer revision recorded with token counts.",
    )
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/data_audit/helpsteer2.json"),
        help="Aggregate JSON report path.",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        help=(
            "Optional normalized JSONL path. Keep it under data/processed/ or "
            "another ignored artifact directory."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pointwise_rows, preference_rows = _load_helpsteer2(
        args.dataset,
        revision=args.revision,
        preference_data_dir=args.preference_data_dir,
    )
    normalization = normalize_helpsteer2_pairs(
        pointwise_rows,
        preference_rows,
        strict=False,
    )
    pairs = assign_prompt_splits(
        normalization.pairs,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )

    tokenizer_metadata: dict[str, object] | None = None
    if args.tokenizer:
        tokenizer = _load_tokenizer(
            args.tokenizer,
            revision=args.tokenizer_revision,
        )
        pairs = attach_token_counts(
            pairs,
            tokenizer,
            batch_size=args.tokenizer_batch_size,
        )
        tokenizer_metadata = {
            "name": args.tokenizer,
            "revision": args.tokenizer_revision,
            "class": type(tokenizer).__name__,
            "vocabulary_size": getattr(tokenizer, "vocab_size", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "response_special_tokens": False,
        }

    pairs_artifact = (
        _write_pairs(args.pairs_output, pairs)
        if args.pairs_output is not None
        else None
    )
    pair_audit = audit_pairs(pairs)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": args.dataset,
            "revision": args.revision,
            "preference_data_dir": args.preference_data_dir,
        },
        "experimental_split": {
            "unit": "prompt_id",
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
        },
        "tokenizer": tokenizer_metadata,
        "software": {
            "python": platform.python_version(),
            "datasets": version("datasets"),
            "transformers": (
                version("transformers") if tokenizer_metadata is not None else None
            ),
            "tokenizers": (
                version("tokenizers") if tokenizer_metadata is not None else None
            ),
        },
        "alignment": normalization.report.to_dict(),
        "pairs": pair_audit.to_dict(),
        "pairs_artifact": pairs_artifact,
    }
    _write_json(args.output, report)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nAudit report: {args.output}")
    if args.pairs_output is not None:
        print(f"Normalized pairs: {args.pairs_output}")

    return 0 if normalization.report.is_valid and pair_audit.is_valid else 1


def _load_helpsteer2(
    dataset_name: str,
    *,
    revision: str,
    preference_data_dir: str,
) -> tuple[Iterable[Mapping[str, object]], Iterable[Mapping[str, object]]]:
    try:
        from datasets import DatasetDict, load_dataset
    except ModuleNotFoundError:
        sys.exit(
            "The optional 'datasets' package is required. "
            "Run `uv sync --extra data` and retry."
        )

    pointwise = load_dataset(dataset_name, revision=revision)
    if not isinstance(pointwise, DatasetDict):
        raise TypeError("the pointwise dataset must provide named splits")
    preferences = load_dataset(
        dataset_name,
        data_dir=preference_data_dir,
        revision=revision,
        split="train",
    )
    return _pointwise_rows(pointwise), preferences


def _pointwise_rows(
    dataset: Mapping[str, Iterable[Mapping[str, Any]]],
) -> Iterator[dict[str, Any]]:
    ordered_splits = [
        *[name for name in ("train", "validation", "test") if name in dataset],
        *sorted(
            name
            for name in dataset
            if name not in {"train", "validation", "test"}
        ),
    ]
    for split_name in ordered_splits:
        for row in dataset[split_name]:
            yield {**row, "_source_split": split_name}


def _load_tokenizer(name: str, *, revision: str) -> object:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        sys.exit(
            "Token-length diagnostics require Transformers. "
            "Run `uv sync --extra tokenizer` and retry."
        )
    return AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        use_fast=True,
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_pairs(
    path: Path,
    pairs: Iterable[NormalizedPair],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    row_count = 0
    with path.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            line = json.dumps(pair.to_dict(), ensure_ascii=False) + "\n"
            stream.write(line)
            digest.update(line.encode())
            row_count += 1
    return {
        "path": str(path),
        "rows": row_count,
        "sha256": digest.hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
