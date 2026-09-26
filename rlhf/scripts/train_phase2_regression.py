#!/usr/bin/env python3
"""Fine-tune Llama 3.1 to predict HelpSteer2 helpfulness."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_FLAX", "0")

sys.path.insert(0, str(Path(__file__).parent))
from train_phase2_bt import trainer_class, training_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


class RegressionDataset:
    def __init__(self, path: Path, use_fair_reweighing: bool):
        with path.open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        for row in self.rows:
            row["sample_weight"] = (
                float(row["fair_reweighing_weight"])
                if use_fair_reweighing
                else 1.0
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class RegressionCollator:
    def __init__(self, tokenizer: Any, max_length: int, pad_multiple: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_multiple = pad_multiple

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        texts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for row in rows
        ]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            pad_to_multiple_of=self.pad_multiple,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if lengths != [row["sequence_length"] for row in rows]:
            raise ValueError("runtime tokenization differs from the prepared data")
        if max(lengths) > self.max_length:
            raise ValueError("a response exceeds the configured maximum length")
        return {
            **encoded,
            "labels": torch.tensor([row["helpfulness"] for row in rows]),
            "sample_weight": torch.tensor([row["sample_weight"] for row in rows]),
        }


def regression_trainer_class(transformers: Any) -> type[Any]:
    BaseTrainer = trainer_class(transformers)

    class RegressionTrainer(BaseTrainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            labels = inputs.pop("labels").reshape(-1)
            weights = inputs.pop("sample_weight").reshape(-1)
            output = model(**inputs)
            predictions = output.logits.reshape(-1).float()
            loss = (weights.to(predictions) * (predictions - labels.to(predictions)) ** 2).mean()
            return (loss, {"logits": predictions}) if return_outputs else loss

    return RegressionTrainer


def regression_metrics(prediction: Any) -> dict[str, float]:
    from scipy.stats import spearmanr

    predicted = np.asarray(prediction.predictions).reshape(-1)
    labels = np.asarray(prediction.label_ids).reshape(-1)
    residual = predicted - labels
    return {
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(labels, predicted).statistic),
    }


def main() -> None:
    args = parse_args()
    for target, source in {
        "RANK": "SLURM_PROCID",
        "LOCAL_RANK": "SLURM_LOCALID",
        "WORLD_SIZE": "SLURM_NTASKS",
    }.items():
        if source in os.environ:
            os.environ.setdefault(target, os.environ[source])

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    training = config["training"]
    distributed = config["distributed"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["FSDP_STATE_DICT_TYPE"] = str(distributed["state_dict_type"]).upper()

    import torch
    import transformers

    accumulation = int(training["global_batch_size"]) // (
        int(training["micro_batch_size"]) * world_size
    )
    arguments = training_arguments(transformers, config, accumulation, local_rank)
    transformers.set_seed(int(training["seed"]))

    use_weights = training.get("treatment") == "fair_reweighing"
    root = Path(data["root"])
    train = RegressionDataset(root / data["train_file"], use_weights)
    validation = RegressionDataset(root / data["validation_file"], False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        data["tokenizer_name"], revision=data["tokenizer_revision"], use_fast=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_config = config["model"]
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        num_labels=1,
        torch_dtype=getattr(torch, model_config["torch_dtype"]),
        low_cpu_mem_usage=True,
        attn_implementation=model_config["attention_implementation"],
        local_files_only=model_config.get("cache_policy") == "local_only",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.problem_type = "regression"

    Trainer = regression_trainer_class(transformers)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=RegressionCollator(
            tokenizer, int(data["max_length"]), int(data["pad_to_multiple_of"])
        ),
        compute_metrics=regression_metrics,
        processing_class=tokenizer,
    )
    trainer.label_names = ["labels"]
    if distributed.get("sync_each_batch") and accumulation > 1:
        trainer.accelerator.no_sync = lambda *args, **kwargs: contextlib.nullcontext()

    train_result = trainer.train()
    validation_result = trainer.evaluate(metric_key_prefix="final_validation")
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("final_validation", validation_result)

    if trainer.is_world_process_zero():
        output = Path(config["output"]["directory"])
        target = output / f"checkpoint-{training['target_checkpoint_step']}"
        (output / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config": str(args.config),
                    "treatment": training.get("treatment", "none"),
                    "target_checkpoint": str(target),
                    "train": train_result.metrics,
                    "validation": validation_result,
                },
                indent=2,
                default=float,
            )
            + "\n"
        )
        print(f"target checkpoint: {target}")


if __name__ == "__main__":
    main()
