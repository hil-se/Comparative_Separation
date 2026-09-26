# Comparative Separation

Replication package for *Comparative Separation: Evaluating Separation on
Comparative Judgment Test Data*.

| Directory | Experiment | Entry point |
|---|---|---|
| [simulations](simulations/) | Synthetic classifiers | `python test0.py` through `test3.py` |
| [classification](classification/) | COMPAS and German Credit | `python src/classification.py` (see working directory instructions) |
| [regression](regression/) | Jira story points | `python src/regression.py` (see working directory instructions) |
| [rlhf](rlhf/) | HelpSteer2 reward models | `python rlhf/reproduce_table.py` |

Use Python 3.11 and install `pip install -r requirements.txt`. For recalculating
the HelpSteer2 table from saved predictions, only NumPy, SciPy, and scikit-learn
are needed; no GPU or model download is required.

Each section documents its data, commands, and outputs. Saved classification
and regression tables are included. The HelpSteer2 table is recalculated from
the eight submitted models' response and pair scores, with checkpoint provenance.
Refitting stochastic models is distinct from recalculating a saved result.

`real/`, `simulation/`, and root `cp.py` are retained for compatibility with
earlier links. The four directories above are the replication entry points.
The Jira experiment uses the historical in-sample separation estimator; the
HelpSteer2 experiment uses prompt-grouped cross-fitting. These are not interchangeable.
