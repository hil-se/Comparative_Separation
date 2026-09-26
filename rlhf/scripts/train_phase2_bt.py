#!/usr/bin/env python3
"""Fine-tune one HelpSteer2 Bradley-Terry reward model with FSDP."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import tomllib
from typing import Any

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_FLAX", "0")

from fair_rlhf.phase2_bt import (
    configure_phase2_chat_template,
    effective_gradient_accumulation_steps,
    pairwise_reward_loss,
    pairwise_reward_loss_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class PairDataset:
    def __init__(self, path: Path):
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class PairCollator:
    """Render each chosen/rejected pair as two adjacent model inputs."""

    def __init__(self, tokenizer: Any, max_length: int, pad_multiple: int, weight_field: str | None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_multiple = pad_multiple
        self.weight_field = weight_field

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        texts, expected_lengths = [], []
        for row in examples:
            for side in ("chosen", "rejected"):
                messages = [
                    *row["context"],
                    {"role": "assistant", "content": row[side]},
                ]
                texts.append(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                )
                expected_lengths.append(int(row[f"{side}_sequence_length"]))

        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            pad_to_multiple_of=self.pad_multiple,
            return_tensors="pt",
        )
        actual_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if actual_lengths != expected_lengths:
            raise ValueError("runtime tokenization differs from the prepared data")
        if max(actual_lengths) > self.max_length:
            raise ValueError("a pair exceeds the configured maximum length")

        batch = {
            **encoded,
            "preference_strength": torch.tensor(
                [row["preference_strength"] for row in examples], dtype=torch.float32
            ),
        }
        if self.weight_field:
            batch["pair_weight"] = torch.tensor(
                [row[self.weight_field] for row in examples], dtype=torch.float32
            )
        return batch


def trainer_class(transformers: Any) -> type[Any]:
    class PairwiseTrainer(transformers.Trainer):
        pairwise_loss_name = "binary_bradley_terry"
        use_pair_weights = False
        sequential_sampler = False

        def _get_train_sampler(self) -> Any:
            if self.sequential_sampler:
                from torch.utils.data import SequentialSampler

                return SequentialSampler(self.train_dataset)
            return super()._get_train_sampler()

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            strength = inputs.pop("preference_strength")
            weight = inputs.pop("pair_weight", None)
            output = model(**inputs)
            rewards = output.logits.reshape(-1, 2)
            losses = pairwise_reward_loss(
                rewards[:, 0],
                rewards[:, 1],
                strength,
                loss=self.pairwise_loss_name,
                reduction="none",
            )
            loss = (losses * weight.to(losses)).mean() if self.use_pair_weights else losses.mean()
            return (loss, {"logits": rewards}) if return_outputs else loss

        def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
            """Write model-only FSDP shards without gathering 70B on rank zero."""

            if not self.is_fsdp_enabled:
                return super().save_model(output_dir, _internal_call)
            state_type = str(self.accelerator.state.fsdp_plugin.state_dict_type)
            if "SHARDED_STATE_DICT" not in state_type:
                return super().save_model(output_dir, _internal_call)

            import torch
            from accelerate.utils import save_fsdp_model

            output_dir = output_dir or self.args.output_dir
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model,
                output_dir,
            )
            if self.args.should_save:
                path = Path(output_dir)
                model = self.accelerator.unwrap_model(self.model)
                model.config.save_pretrained(path)
                self.processing_class.save_pretrained(path)
                torch.save(self.args, path / "training_args.bin")
            self.accelerator.wait_for_everyone()

    return PairwiseTrainer


def training_arguments(
    transformers: Any,
    config: dict[str, Any],
    gradient_accumulation: int,
    local_rank: int,
) -> Any:
    training = config["training"]
    distributed = config["distributed"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return transformers.TrainingArguments(
        output_dir=str(config["output"]["directory"]),
        overwrite_output_dir=bool(config["output"]["overwrite"]),
        run_name=str(config["study"]["name"]),
        do_train=True,
        do_eval=True,
        num_train_epochs=float(training["epochs"]),
        max_steps=int(training.get("max_steps", -1)),
        per_device_train_batch_size=int(training["micro_batch_size"]),
        per_device_eval_batch_size=int(training["eval_micro_batch_size"]),
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        adam_beta1=float(training["adam_beta1"]),
        adam_beta2=float(training["adam_beta2"]),
        adam_epsilon=float(training["adam_epsilon"]),
        optim=str(training["optimizer"]),
        lr_scheduler_type=str(training["lr_scheduler"]),
        warmup_steps=int(training["warmup_steps"]),
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=True,
        bf16_full_eval=True,
        tf32=bool(training["tf32"]),
        eval_strategy="steps",
        eval_steps=int(training["eval_steps"]),
        save_strategy="steps",
        save_steps=1 if "save_at_steps" in training else int(training["save_steps"]),
        save_only_model=True,
        logging_steps=int(training["logging_steps"]),
        report_to=[],
        remove_unused_columns=False,
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        local_rank=local_rank,
        fsdp="full_shard auto_wrap" if world_size > 1 else "",
        fsdp_config={
            "transformer_layer_cls_to_wrap": [distributed["transformer_layer_class"]],
            "backward_prefetch": distributed["backward_prefetch"],
            "forward_prefetch": distributed["forward_prefetch"],
            "limit_all_gathers": True,
            "use_orig_params": distributed["use_orig_params"],
            "sync_module_states": True,
            "cpu_ram_efficient_loading": True,
            "activation_checkpointing": distributed["activation_checkpointing"],
            "state_dict_type": str(distributed["state_dict_type"]).upper(),
        }
        if world_size > 1
        else {},
    )


def validation_metrics(loss_name: str):
    def compute(prediction: Any) -> dict[str, float]:
        rewards = np.asarray(prediction.predictions).reshape(-1, 2)
        strength = np.asarray(prediction.label_ids).reshape(-1)
        margin = rewards[:, 0] - rewards[:, 1]
        values = pairwise_reward_loss_values(
            rewards[:, 0], rewards[:, 1], strength, loss=loss_name
        )
        return {
            "pairwise_accuracy": float((margin > 0).mean()),
            "mean_reward_margin": float(margin.mean()),
            f"{loss_name}_loss": float(values.mean()),
        }

    return compute


def scheduled_save_callback(transformers: Any, steps: list[int] | None) -> Any:
    """Keep the requested save schedule and always save the final step."""

    class ScheduledSaveCallback(transformers.TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            if steps is not None:
                control.should_save = state.global_step in steps
            if state.global_step == state.max_steps:
                control.should_save = True
            return control

    return ScheduledSaveCallback()


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

    gradient_accumulation = effective_gradient_accumulation_steps(
        global_batch_size=int(training["global_batch_size"]),
        micro_batch_size=int(training["micro_batch_size"]),
        world_size=world_size,
    )
    arguments = training_arguments(
        transformers, config, gradient_accumulation, local_rank
    )
    transformers.set_seed(int(training["seed"]))

    root = Path(data["root"])
    train = PairDataset(root / data["train_file"])
    validation = PairDataset(root / data["validation_file"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        data["tokenizer_name"], revision=data["tokenizer_revision"], use_fast=True
    )
    configure_phase2_chat_template(
        tokenizer, serialization_profile="tokenizer_native"
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
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.problem_type = "regression"

    weight_field = data.get("sample_weight_field")
    Trainer = trainer_class(transformers)
    if training.get("keep_lowest_validation"):
        from topk_validation import topk_validation_callback
        checkpoint_callback = topk_validation_callback(
            transformers, int(training["keep_lowest_validation"]),
            config["output"]["selection_manifest"],
        )
    else:
        checkpoint_callback = scheduled_save_callback(transformers, training.get("save_at_steps"))
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=PairCollator(
            tokenizer, int(data["max_length"]), int(data["pad_to_multiple_of"]), weight_field
        ),
        compute_metrics=validation_metrics(training["loss"]),
        processing_class=tokenizer,
        callbacks=[checkpoint_callback],
    )
    trainer.pairwise_loss_name = training["loss"]
    trainer.use_pair_weights = weight_field is not None
    trainer.sequential_sampler = data.get("train_sampler") == "sequential"
    trainer.label_names = ["preference_strength"]

    # FSDP cannot retain full gradients during accumulation on a 70B model.
    if distributed.get("sync_each_batch") and gradient_accumulation > 1:
        trainer.accelerator.no_sync = lambda *args, **kwargs: contextlib.nullcontext()

    output_dir = Path(config["output"]["directory"])
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "config": str(args.config),
                    "train_pairs": len(train),
                    "validation_pairs": len(validation),
                    "world_size": world_size,
                    "gradient_accumulation_steps": gradient_accumulation,
                },
                indent=2,
            )
            + "\n"
        )

    train_result = trainer.train()
    validation_result = trainer.evaluate(metric_key_prefix="final_validation")
    # Keep the complete validation-loss history even when only the final model
    # checkpoint is materialized.  Paper-style checkpoint selection uses this
    # history to identify the three lowest-loss steps for deterministic reruns.
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("final_validation", validation_result)
    if trainer.is_world_process_zero():
        selected_step = (checkpoint_callback.selected[0]
                         if training.get("keep_lowest_validation") else trainer.state.global_step)
        target = output_dir / f"checkpoint-{selected_step}"
        (output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config": str(args.config),
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
