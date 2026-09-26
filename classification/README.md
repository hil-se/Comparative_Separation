# COMPAS and German Credit

The submitted table is saved in `result/classification.csv`. To rerun it from
the included data:

```sh
cd classification/src
python classification.py
```

The new table is written to `classification/output/classification.csv`, leaving
the saved paper result unchanged. The script uses NumPy seed 0, 1,000 repeated
splits, logistic regression, and no treatment, FairBalance, or Reweighing.
`exp.py` splits within protected-attribute/label groups; `preprocessor.py`
implements training-data treatments. `metrics.py` implements separation and
comparative separation. Table entries are rejection frequencies at alpha=0.05.

Data sources: [COMPAS](https://www.kaggle.com/datasets/danofer/compass) and
[German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
The included copies are the inputs used by the experiment.
