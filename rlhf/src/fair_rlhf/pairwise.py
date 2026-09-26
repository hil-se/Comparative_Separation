"""Pairwise length-bias treatments and Comparative Separation metrics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class ComparativeSeparationResult:
    tpr_preferred_longer: float
    tpr_preferred_shorter: float
    signed_gap: float
    absolute_gap: float
    z_statistic: float
    p_value: float
    n_preferred_longer: int
    n_preferred_shorter: int
    n_equal_length: int


@dataclass(frozen=True)
class ConditionalReweighingResult:
    weights: NDArray[np.float64]
    quantile_boundaries: tuple[float, ...]
    bin_counts: tuple[int, ...]
    requested_bins: int


def _vectors(left: ArrayLike, right: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if len(left) != len(right):
        raise ValueError("pair arrays must have the same length")
    return left, right


def _labels(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=int).reshape(-1)


def length_relation(left_token_count: ArrayLike, right_token_count: ArrayLike) -> np.ndarray:
    """Return -1, 0, or +1 according to which displayed response is longer."""

    left, right = _vectors(left_token_count, right_token_count)
    return np.sign(left - right).astype(int)


def preferred_length_relation(
    left_token_count: ArrayLike,
    right_token_count: ArrayLike,
    preference_label: ArrayLike,
) -> np.ndarray:
    """Return +1 when the preferred response is longer and -1 when shorter."""

    relation = length_relation(left_token_count, right_token_count)
    return relation * _labels(preference_label)


def _cell_weight_lookup(
    relation: np.ndarray,
    labels: np.ndarray,
    smoothing: float,
) -> dict[tuple[int, int], float]:
    """Compute P(D)P(Y)/P(D,Y) once for every observed D/Y cell."""

    relations = np.unique(relation)
    outcomes = np.array([-1, 1])
    counts = np.full((len(relations), 2), smoothing, dtype=float)
    relation_index = {value: index for index, value in enumerate(relations)}
    outcome_index = {-1: 0, 1: 1}

    for d, y in zip(relation, labels, strict=True):
        counts[relation_index[d], outcome_index[y]] += 1

    joint = counts / counts.sum()
    weights = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :] / joint
    return {
        (int(d), int(y)): float(weights[relation_index[d], outcome_index[y]])
        for d in relations
        for y in outcomes
    }


def _row_weights(
    relation: np.ndarray,
    labels: np.ndarray,
    lookup: dict[tuple[int, int], float],
) -> np.ndarray:
    values = np.array([lookup[(int(d), int(y))] for d, y in zip(relation, labels)])
    return values / values.mean()


def pairwise_reweighing_weights(
    left_token_count: ArrayLike,
    right_token_count: ArrayLike,
    preference_label: ArrayLike,
    *,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Original Comparative FairReweighing: w(D,Y)=P(D)P(Y)/P(D,Y)."""

    relation = length_relation(left_token_count, right_token_count)
    labels = _labels(preference_label)
    return _row_weights(relation, labels, _cell_weight_lookup(relation, labels, smoothing))


def reverse_pairwise_reweighing_weights(
    left_token_count: ArrayLike,
    right_token_count: ArrayLike,
    preference_label: ArrayLike,
    *,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Fit the same weights after adding each logical reverse pair (-D,-Y)."""

    relation = length_relation(left_token_count, right_token_count)
    labels = _labels(preference_label)
    symmetric_relation = np.concatenate((relation, -relation))
    symmetric_labels = np.concatenate((labels, -labels))
    lookup = _cell_weight_lookup(symmetric_relation, symmetric_labels, smoothing)
    return _row_weights(relation, labels, lookup)


def conditional_pairwise_reweighing_weights(
    left_token_count: ArrayLike,
    right_token_count: ArrayLike,
    preference_label: ArrayLike,
    *,
    quantile_bins: int = 4,
    smoothing: float = 0.5,
) -> ConditionalReweighingResult:
    """Estimate Comparative FairReweighing separately within length-gap bins."""

    left, right = _vectors(left_token_count, right_token_count)
    labels = _labels(preference_label)
    relation = np.sign(left - right).astype(int)
    gap = np.abs(left - right)
    boundaries = np.unique(np.quantile(gap, np.arange(1, quantile_bins) / quantile_bins))
    bin_id = np.searchsorted(boundaries, gap, side="right")

    weights = np.empty(len(labels))
    counts = []
    for current_bin in range(len(boundaries) + 1):
        rows = bin_id == current_bin
        counts.append(int(rows.sum()))
        lookup = _cell_weight_lookup(relation[rows], labels[rows], smoothing)
        weights[rows] = _row_weights(relation[rows], labels[rows], lookup)

    weights /= weights.mean()
    return ConditionalReweighingResult(
        weights=weights,
        quantile_boundaries=tuple(float(value) for value in boundaries),
        bin_counts=tuple(counts),
        requested_bins=quantile_bins,
    )


def comparative_separation_from_scores(
    left_score: ArrayLike,
    right_score: ArrayLike,
    preference_label: ArrayLike,
    left_token_count: ArrayLike,
    right_token_count: ArrayLike,
) -> ComparativeSeparationResult:
    """Compare accuracy when the preferred response is longer versus shorter."""

    left, right = _vectors(left_score, right_score)
    labels = _labels(preference_label)
    group = preferred_length_relation(left_token_count, right_token_count, labels)
    correct = np.sign(left - right).astype(int) == labels
    longer, shorter, equal = group == 1, group == -1, group == 0

    long_accuracy = float(correct[longer].mean())
    short_accuracy = float(correct[shorter].mean())
    gap = long_accuracy - short_accuracy
    variance = (
        long_accuracy * (1 - long_accuracy) / longer.sum()
        + short_accuracy * (1 - short_accuracy) / shorter.sum()
    )
    if variance == 0:
        z = 0.0 if gap == 0 else float(np.sign(gap) * np.inf)
    else:
        z = gap / np.sqrt(variance)
    p = float(2 * norm.sf(abs(z)))

    return ComparativeSeparationResult(
        tpr_preferred_longer=long_accuracy,
        tpr_preferred_shorter=short_accuracy,
        signed_gap=gap,
        absolute_gap=abs(gap),
        z_statistic=float(z),
        p_value=p,
        n_preferred_longer=int(longer.sum()),
        n_preferred_shorter=int(shorter.sum()),
        n_equal_length=int(equal.sum()),
    )
