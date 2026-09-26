# Jira story-point regression

The submitted results are in `result/regression.csv`. To refit the models:

```sh
cd regression/src
python regression.py
```

This downloads `sentence-transformers/all-MiniLM-L6-v2` on first use. A linear
Keras regressor is trained on the text embeddings with MAE loss, Adam, batch
size 32, and 600 epochs. FairReweighing uses training-data densities. The
included `split_mark` selects the test set; the best training-loss checkpoint
is loaded. Test loss is logged, not used by the checkpoint/LR callbacks.

New results go to `output/regression.csv`; checkpoints go to `src/checkpoint/`.
The saved paper table is not overwritten. NumPy's pair-sampling seed is 0;
the historical fit did not fix TensorFlow's seed, so retraining is stochastic.
`Isep` is the historical in-sample estimator, not cross-fitted Csep. Both
cross-group (`pc`, `dc`) and within-group (`pw`, `dw`) contrasts are reported.

The included `jirasoftware_filtered.csv` comes from the
[GPT2SP marked data](https://github.com/awsm-research/gpt2sp/tree/main/sp_dataset/marked_data).
The sensitive attribute is the supplied `is_internal` indicator and the target
is `storypoint`; the existing coding and split are used without relabeling.
