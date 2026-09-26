"""Keep the three lowest unweighted validation-loss checkpoints for an opted-in run."""

import json
import math
from pathlib import Path
import shutil


def topk_validation_callback(transformers, keep, manifest_path):
    class TopKValidation(transformers.TrainerCallback):
        def __init__(self):
            self.losses = {}
            self.selected = []
            self.saved = set()

        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = False
            if state.global_step == state.max_steps:
                control.should_evaluate = True
            return control

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if "eval_loss" not in metrics:
                return control
            loss = float(metrics["eval_loss"])
            if not math.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            self.losses[state.global_step] = loss
            self.selected = sorted(self.losses, key=lambda s: (self.losses[s], s))[:keep]
            control.should_save = state.global_step in self.selected and state.global_step not in self.saved
            return control

        def on_save(self, args, state, control, **kwargs):
            self.saved.add(state.global_step)
            if state.is_world_process_zero:
                # Only remove checkpoints created by this callback in this run.
                for step in self.saved - set(self.selected):
                    shutil.rmtree(Path(args.output_dir) / f"checkpoint-{step}")
                document = {
                    "metric": "eval_loss", "validation_weight_policy": "unit_weight",
                    "selection": "lowest_three_no_extra_final", "final_step_seen": state.global_step,
                    "checkpoints": [{"step": s, "validation_loss": self.losses[s],
                                     "path": str(Path(args.output_dir) / f"checkpoint-{s}")}
                                    for s in self.selected],
                }
                for path in [Path(args.output_dir) / "selected_checkpoints.json", Path(manifest_path)]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(document, indent=2) + "\n")
            self.saved.intersection_update(self.selected)
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
            return control

    return TopKValidation()
