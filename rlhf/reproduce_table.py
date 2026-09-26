"""Recalculate the submitted HelpSteer2 table from saved model predictions."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fair_rlhf.cp import CP
from fair_rlhf.metrics import cross_fitted_csep, comparative_separation_cluster_inference


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def column(records, key):
    return np.asarray([row[key] for row in records])


def evaluate(model):
    directory = ROOT / "scores" / model["id"]
    responses = rows(directory / "response_scores.jsonl")
    pairs = rows(directory / "pair_scores.jsonl")
    left, right = (column(pairs, key) for key in ("left_score", "right_score"))
    labels = column(pairs, "preference_label")
    left_len, right_len = (column(pairs, key) for key in ("left_token_count", "right_token_count"))
    groups = np.sign(left_len - right_len)
    keep = groups != 0
    csep = cross_fitted_csep(
        column(responses, "helpfulness"), column(responses, "score"),
        column(responses, "response_token_count"),
        groups=column(responses, "prompt_id"), seed=1234, n_splits=10,
    )
    cluster = comparative_separation_cluster_inference(
        left, right, labels, left_len, right_len, column(pairs, "prompt_id"),
        repetitions=2000, seed=1234,
    )
    reference = CP(labels[keep], (left - right)[keep]).comparative_separation(groups[keep])
    result = {
        "treatment": model["label"],
        "preference_accuracy": float(np.mean(labels * (left - right) > 0)),
        "csep_xfit": csep.value,
        "p_cluster": cluster.randomization_p_value,
        "d": reference.dc,
        "responses": len(responses), "preference_pairs": len(pairs),
        "unequal_length_pairs": int(keep.sum()),
    }
    for key, expected in model["expected"].items():
        if not np.isclose(result[key], expected, rtol=0, atol=1e-10):
            raise AssertionError(f"{model['id']} {key}: {result[key]} != {expected}")
    return result


if __name__ == "__main__":
    manifest = json.loads((ROOT / "manifest.json").read_text())
    results = [evaluate(model) for model in manifest["models"]]
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    with (output / "helpsteer2_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for row in results:
        print(f"{row['treatment']:28} {100*row['preference_accuracy']:6.2f}%  "
              f"{row['csep_xfit']:.4f}  {row['p_cluster']:.4f}  {row['d']:+.4f}")
    print(f"Saved {output / 'helpsteer2_table.csv'}; all eight rows match the archived metrics.")
