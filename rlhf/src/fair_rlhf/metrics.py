"""Pointwise and prompt-cluster fairness metrics for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .pairwise import preferred_length_relation


EPS = 1e-12


@dataclass(frozen=True)
class CsepResult:
    """Cross-fitted conditional mutual-information estimate."""

    value: float
    raw_value: float
    folds: int
    observations: int
    prompt_clusters: int


@dataclass(frozen=True)
class ClusterInferenceResult:
    """Prompt-cluster uncertainty for the comparative-separation gap."""

    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    randomization_p_value: float
    repetitions: int
    valid_bootstrap_repetitions: int
    valid_randomization_repetitions: int
    prompt_clusters: int


def cross_fitted_csep(
    quality: ArrayLike,
    reward: ArrayLike,
    token_count: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    seed: int = 0,
    n_splits: int = 10,
) -> CsepResult:
    """Estimate ``I(reward; token_count | quality)`` out of sample.

    This is the Gaussian conditional-density estimator used by the source
    FairReweighing project. Each held-out log-density ratio compares a model of
    token count from ``(quality, reward)`` with a model using quality alone.
    When prompt IDs are supplied, all responses from a prompt remain in the
    same cross-fitting fold.
    """

    quality_values = _finite_vector(quality, "quality")
    reward_values = _finite_vector(reward, "reward")
    token_values = _finite_vector(token_count, "token_count")
    _check_lengths(quality_values, reward_values, token_values)
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")

    group_values = _groups(groups, len(quality_values))
    cluster_count = len(np.unique(group_values))
    if len(np.unique(token_values)) <= 1:
        return CsepResult(
            value=0.0,
            raw_value=0.0,
            folds=min(n_splits, cluster_count),
            observations=len(token_values),
            prompt_clusters=cluster_count,
        )

    joint = np.column_stack((quality_values, reward_values))
    margin = quality_values.reshape(-1, 1)
    log_joint: list[float] = []
    log_margin: list[float] = []
    folds = _group_folds(group_values, seed=seed, n_splits=n_splits)

    for train, test in folds:
        joint_model = LinearRegression().fit(joint[train], token_values[train])
        margin_model = LinearRegression().fit(margin[train], token_values[train])

        joint_train_prediction = joint_model.predict(joint[train])
        margin_train_prediction = margin_model.predict(margin[train])
        joint_scale = max(
            float(np.std(token_values[train] - joint_train_prediction)),
            EPS,
        )
        margin_scale = max(
            float(np.std(token_values[train] - margin_train_prediction)),
            EPS,
        )

        log_joint.extend(
            norm.logpdf(
                token_values[test],
                joint_model.predict(joint[test]),
                joint_scale,
            )
        )
        log_margin.extend(
            norm.logpdf(
                token_values[test],
                margin_model.predict(margin[test]),
                margin_scale,
            )
        )

    if not log_joint:
        raise ValueError("Csep cross-fitting produced no held-out observations")
    raw_value = float(
        np.mean(np.asarray(log_joint, dtype=float) - np.asarray(log_margin, dtype=float))
    )
    return CsepResult(
        value=max(0.0, raw_value),
        raw_value=raw_value,
        folds=len(folds),
        observations=len(token_values),
        prompt_clusters=cluster_count,
    )


def comparative_separation_cluster_inference(
    left_score: ArrayLike,
    right_score: ArrayLike,
    preference_label: ArrayLike,
    left_token_count: ArrayLike,
    right_token_count: ArrayLike,
    prompt_ids: ArrayLike,
    *,
    repetitions: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ClusterInferenceResult:
    """Bootstrap and randomize the separation gap by prompt cluster.

    Bootstrap replicates sample prompt clusters with replacement. The
    randomization test flips the longer/shorter group for every observation in
    a prompt together, preserving within-prompt dependence.
    """

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    left = _finite_vector(left_score, "left_score")
    right = _finite_vector(right_score, "right_score")
    labels = np.asarray(preference_label, dtype=int).reshape(-1)
    left_length = _finite_vector(left_token_count, "left_token_count")
    right_length = _finite_vector(right_token_count, "right_token_count")
    prompt_values = np.asarray(prompt_ids).astype(str).reshape(-1)
    _check_lengths(left, right, labels, left_length, right_length, prompt_values)
    if not np.isin(labels, (-1, 1)).all():
        raise ValueError("preference_label must contain only -1 and +1")
    if np.any(prompt_values == ""):
        raise ValueError("prompt_ids must not contain empty values")

    length_group = preferred_length_relation(left_length, right_length, labels)
    keep = length_group != 0
    correct = (np.sign(left - right).astype(int) == labels)[keep].astype(float)
    length_group = length_group[keep]
    prompt_values = prompt_values[keep]
    observed = _difference_in_means(correct, length_group)

    clusters, inverse = np.unique(prompt_values, return_inverse=True)
    cluster_rows = tuple(np.flatnonzero(inverse == index) for index in range(len(clusters)))
    bootstrap_rng, randomization_rng = (
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(2)
    )

    bootstrap_values: list[float] = []
    for _ in range(repetitions):
        selected = bootstrap_rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([cluster_rows[index] for index in selected])
        value = _difference_in_means_or_nan(correct[rows], length_group[rows])
        if np.isfinite(value):
            bootstrap_values.append(value)

    randomized_values: list[float] = []
    for _ in range(repetitions):
        flips = randomization_rng.choice((-1, 1), size=len(clusters))
        randomized_group = length_group * flips[inverse]
        value = _difference_in_means_or_nan(correct, randomized_group)
        if np.isfinite(value):
            randomized_values.append(value)

    if not bootstrap_values:
        raise ValueError("no valid prompt-cluster bootstrap replicate was produced")
    if not randomized_values:
        raise ValueError("no valid prompt-cluster randomization replicate was produced")

    bootstrap_array = np.asarray(bootstrap_values, dtype=float)
    randomization_array = np.asarray(randomized_values, dtype=float)
    lower, upper = np.quantile(bootstrap_array, (alpha / 2, 1 - alpha / 2))
    p_value = (
        1.0 + np.count_nonzero(np.abs(randomization_array) >= abs(observed))
    ) / (len(randomization_array) + 1.0)
    return ClusterInferenceResult(
        bootstrap_ci_lower=float(lower),
        bootstrap_ci_upper=float(upper),
        randomization_p_value=float(p_value),
        repetitions=repetitions,
        valid_bootstrap_repetitions=len(bootstrap_values),
        valid_randomization_repetitions=len(randomized_values),
        prompt_clusters=len(clusters),
    )


def _group_folds(
    groups: NDArray[np.str_],
    *,
    seed: int,
    n_splits: int,
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
    unique_groups = np.unique(groups)
    k = min(n_splits, len(unique_groups))
    if k < 2:
        raise ValueError("cross-fitting requires at least two prompt clusters")

    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    result: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for train_groups, test_groups in splitter.split(unique_groups):
        train = np.flatnonzero(np.isin(groups, unique_groups[train_groups]))
        test = np.flatnonzero(np.isin(groups, unique_groups[test_groups]))
        result.append((train, test))
    return tuple(result)


def _difference_in_means(correct: NDArray[np.float64], group: NDArray[np.int64]) -> float:
    value = _difference_in_means_or_nan(correct, group)
    if not np.isfinite(value):
        raise ValueError(
            "comparative separation requires both longer-winner and "
            "shorter-winner pairs"
        )
    return float(value)


def _difference_in_means_or_nan(
    correct: NDArray[np.float64],
    group: NDArray[np.int64],
) -> float:
    longer = group == 1
    shorter = group == -1
    if not longer.any() or not shorter.any():
        return float("nan")
    return float(correct[longer].mean() - correct[shorter].mean())


def _finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _groups(values: ArrayLike | None, expected_length: int) -> NDArray[np.str_]:
    if values is None:
        return np.arange(expected_length).astype(str)
    result = np.asarray(values).astype(str).reshape(-1)
    if len(result) != expected_length:
        raise ValueError("groups must match the number of observations")
    if np.any(result == ""):
        raise ValueError("groups must not contain empty values")
    return result


def _check_lengths(*values: NDArray[object]) -> None:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("all metric inputs must have the same length")
