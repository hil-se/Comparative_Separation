# Synthetic experiments

From this directory, run:

```sh
python test0.py
python test1.py
python test2.py
python test3.py
```

The four scripts contain the eight cell probabilities for the four classifiers
in the simulation-distribution table. Each evaluates samples of 1,000 and 2,000
observations with twice as many sampled pairs, over 10,000 repetitions, at
alpha = 0.05. Standard output reports the analytic and sampled quantities used
in the simulation tables. Sampling is stochastic; estimates need not match the
paper's final decimal. `simulation.py` implements the tests and sampling.
