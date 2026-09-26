# HelpSteer2 reward-model experiment

## Reproduce the submitted table (CPU)

From the repository root:

```sh
pip install numpy scipy scikit-learn
python rlhf/reproduce_table.py
```

This recalculates all eight rows from `scores/`, checks them against the archived
metrics in `manifest.json`, and writes `rlhf/output/helpsteer2_table.csv`.
The saved CSV is also included in `result/`. The score files contain IDs,
helpfulness, token counts, labels, and scalar predictions, not model weights.

The common test cohort contains 912 prompts and 1,824 responses. Preference
accuracy uses 695 non-tied preference pairs; comparative separation excludes
three equal-length pairs and uses 692 pairs (416 longer-winner, 276 shorter-winner).
Predicted score ties count as incorrect. Csep uses ten prompt-grouped folds;
the reported p-value uses 2,000 prompt-level randomizations, plus-one correction,
and seed 1234. It is not the asymptotic `pc` returned by `CP`.

## Data and model lineage

The source dataset is `nvidia/HelpSteer2`, revision
`990b2711a36180dd19d9c94b8627844866f8982a`. Preference records are joined to
pointwise ratings. The prompt-disjoint 80/10/10 split uses seed 1, not the
dataset's official split. Regression training has 14,600 responses from 7,299
prompts; validation and test each have 1,824 responses from 912 prompts.
Helpfulness is on the 0–4 scale; response token count is the sensitive attribute.

All eight models use `meta-llama/Llama-3.1-70B-Instruct`, revision
`1605565b47bb9346c5515c34102e054115b4f98b`, with a scalar reward head and full
fine-tuning. The regression models use MSE (weighted for FairReweighing).
Comparative models use binary Bradley–Terry loss; preference strength is not
multiplied into that loss.

| Rows | Training data | Saved-model protocol |
|---|---|---|
| Regression ± FairReweighing | 14,600 responses | 200 optimizer steps, global batch 128 |
| Comparative All (Natural) | 5,705 non-tied pairs | One epoch, validation-selected step 89, global batch 64 |
| Comparative q=0, .25, .5, .75, 1 | 2,324 sampled pairs per arm | Historical 200-step checkpoints, global batch 64 |

These are the submitted rows, not a newly matched training-budget experiment.
`manifest.json` records score hashes, checkpoint identity and test artifact
hashes. `provenance/` retains data audits. Training configs preserve the
historical 200-step composition setting rather than newer one-epoch defaults.
The 75% row uses the corrected exact-test evaluation.

## Rebuild data and train (GPU)

The training source is included under `scripts/` and `src/`. Dataset
normalization is retained from FairRLHF commit `e64eb0e`; the evaluator and
training helpers are copied from the source revision in `manifest.json`.
These scripts require access to the gated Llama model and a configured
distributed GPU environment. No checkpoints or access credentials are included.

From `rlhf/`, after installing the root requirements:

```sh
export PYTHONPATH="$PWD/src"
python scripts/audit_helpsteer2.py \
  --revision 990b2711a36180dd19d9c94b8627844866f8982a \
  --tokenizer meta-llama/Llama-3.1-70B-Instruct \
  --tokenizer-revision 1605565b47bb9346c5515c34102e054115b4f98b \
  --seed 1 --pairs-output data/processed/phase1_pairs/pairs.jsonl
python scripts/prepare_helpsteer2_regression.py
python scripts/prepare_helpsteer2_bt.py
```

Authenticate with Hugging Face and cache the pinned model before training
(the regression configuration uses `local_only`). Use the distributed launcher
appropriate to your cluster; for a single machine with 16 GPUs, for example:

```sh
torchrun --standalone --nproc_per_node=16 scripts/train_phase2_bt.py \
  --config configs/phase2_helpsteer2_bt_llama31_long_050.toml
```

For the other composition arms change `050` to `000`, `025`, `075`, or `100`.
Use `phase2_helpsteer2_bt_llama31_natural_top3.toml` for Natural. For regression,
use `train_phase2_regression.py` and the corresponding regression config.
Multi-node runs need the launcher's node count, ranks, rendezvous, and network
settings; `expected_world_size` is 16. Output paths are relative to this directory.

FSDP checkpoints need HF export before scoring. For a selected checkpoint:

```sh
python scripts/consolidate_phase2_fsdp.py \
  --checkpoint artifacts/checkpoints/RUN/checkpoint-STEP \
  --output-dir artifacts/checkpoints/RUN_hf --expected-numel 69503041536
python scripts/evaluate_phase2_regression.py \
  --checkpoint artifacts/checkpoints/RUN_hf \
  --data-root data/processed/phase2_helpsteer2_regression \
  --output-dir output/RUN --seed 1234 --bootstrap-repetitions 2000
```

Replace `RUN` and `STEP` with the trained model and selected step. Full refitting
is stochastic and hardware/library dependent. The CPU command above is the
direct, verified route for reproducing the submitted numerical table.
