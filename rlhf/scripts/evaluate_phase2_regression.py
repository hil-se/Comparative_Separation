#!/usr/bin/env python3
"""Evaluate a fine-tuned HelpSteer2 scalar reward model.

The evaluator consumes the immutable response-level contract produced by
``prepare_helpsteer2_regression.py``.  It never fits parameters or selects a
checkpoint: scores are produced for the held-out test responses, then the
pointwise and paired verbosity-bias metrics are computed from those scores.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from fair_rlhf.cp import CP
from fair_rlhf.embeddings import stable_response_id
from fair_rlhf.metrics import (
    comparative_separation_cluster_inference,
    cross_fitted_csep,
)
from fair_rlhf.pairwise import comparative_separation_from_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Llama regression checkpoint on HelpSteer2."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-gpu-memory", default="88GiB")
    parser.add_argument("--max-cpu-memory", default="350GiB")
    parser.add_argument("--offload-folder", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--treatment",
        default="none",
        help=(
            "Short run label recorded in metrics.json (for example none, "
            "fair_reweighing, bt_natural, or bt_long_050)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch-size and max-length must be positive")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap-repetitions must be positive")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.treatment) is None:
        raise ValueError("treatment must be a lowercase filesystem-safe label")

    torch, transformers = _dependencies()
    manifest = _load_contract_manifest(args.data_root, args.max_length)
    test_rows = _read_jsonl(args.data_root / "test.jsonl")
    pair_rows = _read_jsonl(args.data_root / "test_pairs.jsonl")
    _validate_test_rows(test_rows, manifest, args.max_length)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("the checkpoint tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    offload_folder = args.offload_folder or args.output_dir / "offload"
    offload_folder.mkdir(parents=True, exist_ok=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: args.max_gpu_memory,
            "cpu": args.max_cpu_memory,
        },
        offload_folder=str(offload_folder),
        attn_implementation="sdpa",
    )
    model.config.problem_type = "regression"
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    scores = _score_rows(
        model,
        tokenizer,
        test_rows,
        torch=torch,
        input_device=input_device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    metrics, pair_scores = _evaluate(
        test_rows,
        pair_rows,
        scores,
        repetitions=args.bootstrap_repetitions,
        alpha=args.alpha,
        seed=args.seed,
    )
    metrics.update(
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint),
            "checkpoint_inventory_sha256": _checkpoint_inventory_sha256(
                args.checkpoint
            ),
            "treatment": args.treatment,
            "data_root": str(args.data_root),
            "manifest_sha256": _file_sha256(args.data_root / "manifest.json"),
            "test_sha256": _file_sha256(args.data_root / "test.jsonl"),
            "test_pairs_sha256": _file_sha256(
                args.data_root / "test_pairs.jsonl"
            ),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "metrics.json", metrics)
    _write_jsonl(args.output_dir / "response_scores.jsonl", _response_score_rows(test_rows, scores))
    _write_jsonl(args.output_dir / "pair_scores.jsonl", pair_scores)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Regression evaluation requires the TIGRIS PyTorch/Transformers environment"
        ) from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run evaluation inside a GPU allocation")
    return torch, transformers


def _load_contract_manifest(root: Path, max_length: int) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "ready":
        raise ValueError(f"{manifest_path} is not ready")
    if int(manifest["sequence_policy"]["max_length"]) != max_length:
        raise ValueError("evaluation max_length does not match the data contract")
    for split in ("test", "test_pairs"):
        path = root / f"{split}.jsonl"
        expected = manifest["artifacts"][split]["sha256"]
        if _file_sha256(path) != expected:
            raise ValueError(f"{split} artifact hash does not match the manifest")
    return manifest


def _validate_test_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    max_length: int,
) -> None:
    if not rows:
        raise ValueError("the test response artifact is empty")
    if len(rows) != int(manifest["split"]["test_responses"]):
        raise ValueError("test response count does not match the manifest")
    for row in rows:
        if row.get("split") != "test":
            raise ValueError("test.jsonl contains a non-test row")
        for key in ("response_id", "prompt_id", "prompt", "response"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"test row has invalid {key}")
        expected_id = stable_response_id(row["prompt"], row["response"])
        if row["response_id"] != expected_id:
            raise ValueError("test response_id is not stable for its text")
        if int(row["sequence_length"]) > max_length:
            raise ValueError("test row exceeds max_length")


def _score_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    input_device: Any,
    batch_size: int,
    max_length: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
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
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
        expected = [int(row["sequence_length"]) for row in batch]
        if lengths != expected:
            raise ValueError("runtime tokenization differs from the data contract")
        if max(lengths) > max_length:
            raise ValueError("a test sequence exceeds max_length")
        model_inputs = {
            name: tensor.to(input_device)
            for name, tensor in encoded.items()
            if name in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            logits = model(**model_inputs).logits.reshape(-1)
        for row, score in zip(batch, logits.float().cpu().tolist(), strict=True):
            scores[str(row["response_id"])] = float(score)
        print(f"scored {min(start + len(batch), len(rows))}/{len(rows)}")
    return scores


def _evaluate(
    test_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    *,
    repetitions: int,
    alpha: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    quality = np.asarray([float(row["helpfulness"]) for row in test_rows], dtype=float)
    reward = np.asarray([scores[str(row["response_id"])] for row in test_rows], dtype=float)
    token_count = np.asarray(
        [int(row["response_token_count"]) for row in test_rows],
        dtype=float,
    )
    prompt_ids = np.asarray([str(row["prompt_id"]) for row in test_rows], dtype=str)
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression

    spearman = spearmanr(quality, reward)
    csep = cross_fitted_csep(
        quality,
        reward,
        token_count,
        groups=prompt_ids,
        seed=seed,
        n_splits=10,
    )
    quality_matrix = quality.reshape(-1, 1)
    reward_residual = reward - LinearRegression().fit(quality_matrix, reward).predict(quality_matrix)
    length_residual = token_count - LinearRegression().fit(quality_matrix, token_count).predict(quality_matrix)
    residual_corr = pearsonr(reward_residual, length_residual)

    left_scores: list[float] = []
    right_scores: list[float] = []
    labels: list[int] = []
    left_lengths: list[int] = []
    right_lengths: list[int] = []
    pair_prompt_ids: list[str] = []
    pair_scores: list[dict[str, Any]] = []
    for row in pair_rows:
        left_id, right_id = _pair_response_ids(row)
        if left_id not in scores or right_id not in scores:
            raise ValueError("test pair references a response missing from test.jsonl")
        label = int(row["preference_label"])
        if label == 0:
            continue
        left_score = float(scores[left_id])
        right_score = float(scores[right_id])
        margin = label * (left_score - right_score)
        left_scores.append(left_score)
        right_scores.append(right_score)
        labels.append(label)
        left_lengths.append(int(row["left_token_count"]))
        right_lengths.append(int(row["right_token_count"]))
        pair_prompt_ids.append(str(row["prompt_id"]))
        pair_scores.append(
            {
                "pair_id": row.get("pair_id"),
                "prompt_id": row["prompt_id"],
                "preference_label": label,
                "left_score": left_score,
                "right_score": right_score,
                "signed_margin": margin,
                "correct": bool(margin > 0),
                "left_token_count": left_lengths[-1],
                "right_token_count": right_lengths[-1],
            }
        )
    if not left_scores:
        raise ValueError("test_pairs.jsonl contains no directional pairs")
    left = np.asarray(left_scores, dtype=float)
    right = np.asarray(right_scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    left_length = np.asarray(left_lengths, dtype=int)
    right_length = np.asarray(right_lengths, dtype=int)
    pair_prompt_array = np.asarray(pair_prompt_ids, dtype=str)
    comparative = comparative_separation_from_scores(
        left, right, label_array, left_length, right_length
    )
    cluster = comparative_separation_cluster_inference(
        left,
        right,
        label_array,
        left_length,
        right_length,
        pair_prompt_array,
        repetitions=repetitions,
        alpha=alpha,
        seed=seed,
    )
    cp_groups = np.sign(left_length - right_length).astype(int)
    cp_keep = cp_groups != 0
    cp_result = CP(
        label_array[cp_keep],
        (left - right)[cp_keep],
    ).comparative_separation(cp_groups[cp_keep])
    return {
        "pointwise": {
            "observations": len(quality),
            "mse": float(np.mean(np.square(reward - quality))),
            "mae": float(np.mean(np.abs(reward - quality))),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": float(spearman.pvalue),
            "csep": asdict(csep),
            "residual_reward_length_correlation": float(residual_corr.statistic),
            "residual_reward_length_p_value": float(residual_corr.pvalue),
        },
        "comparative_separation": {
            **asdict(comparative),
            "cluster_inference": asdict(cluster),
            "reference_cp": {
                **asdict(cp_result),
                "source": "hil-se/Comparative_Separation cp.py@290852b2 with s[i] fix",
                "s_definition": "sign(left_token_count - right_token_count)",
                "within_group_contrast_available": False,
            },
        },
        "pairwise": {
            "pairs": len(labels),
            "accuracy": float(np.mean(label_array * (left - right) > 0)),
            "ties_predicted": int(np.count_nonzero(left == right)),
        },
    }, pair_scores


def _pair_response_ids(row: Mapping[str, Any]) -> tuple[str, str]:
    """Read the stable response IDs stored by the regression data contract."""

    left_id = row.get("left_response_id")
    right_id = row.get("right_response_id")
    if not isinstance(left_id, str) or not left_id:
        raise ValueError("test pair has an invalid left_response_id")
    if not isinstance(right_id, str) or not right_id:
        raise ValueError("test pair has an invalid right_response_id")
    if left_id == right_id:
        raise ValueError("test pair references the same response twice")
    return left_id, right_id


def _response_score_rows(
    rows: Sequence[Mapping[str, Any]], scores: Mapping[str, float]
) -> list[dict[str, Any]]:
    return [
        {
            "response_id": row["response_id"],
            "prompt_id": row["prompt_id"],
            "helpfulness": float(row["helpfulness"]),
            "response_token_count": int(row["response_token_count"]),
            "score": float(scores[str(row["response_id"])]),
        }
        for row in rows
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_inventory_sha256(path: Path) -> str:
    digest = sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(str(item.stat().st_size).encode())
            digest.update(_file_sha256(item).encode())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
