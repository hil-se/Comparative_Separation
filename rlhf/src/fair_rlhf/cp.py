"""Definitions taken from the Comparative Separation reference implementation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CPResult:
    """Reference statistics: cross-group (pc, dc) and within-group (pw, dw)."""

    pc: float
    dc: float
    pw: float
    dw: float


class DensityBalance:
    """Continuous FairReweighing for regression: rho(A)rho(Y)/rho(A,Y)."""

    def __init__(self, model: str = "Neighbor", *, radius: float = 0.5, bandwidth: float = 0.2):
        self.model = model
        self.radius = radius
        self.bandwidth = bandwidth

    def _density(self, values: np.ndarray) -> np.ndarray:
        values = StandardScaler().fit_transform(values)
        if self.model == "Kernel":
            model = KernelDensity(bandwidth=self.bandwidth).fit(values)
            return np.exp(model.score_samples(values))

        model = NearestNeighbors(radius=self.radius).fit(values)
        neighbors = model.radius_neighbors(values, return_distance=False)
        return np.asarray([len(indices) for indices in neighbors], dtype=float)

    def weight(
        self,
        A: ArrayLike,
        y: ArrayLike,
        *,
        treatment: str = "Reweighing",
    ) -> np.ndarray:
        attributes = np.asarray(A, dtype=float)
        if attributes.ndim == 1:
            attributes = attributes[:, None]
        target = np.asarray(y, dtype=float).reshape(-1, 1)

        joint = self._density(np.column_stack((attributes, target)))
        attribute = self._density(attributes)
        outcome = self._density(target)
        formulas = {
            "FairBalanceVariant": 1 / joint,
            "FairBalance": attribute / joint,
            "GroupBalance": outcome / joint,
            "Reweighing": attribute * outcome / joint,
        }
        weights = formulas[treatment]
        return weights / weights.mean()


class CP:
    """Comparative Separation with the corrected ``s[i]`` indexing."""

    def __init__(self, y: ArrayLike, y_pred: ArrayLike):
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    def comparative_separation(self, s: ArrayLike) -> CPResult:
        groups = np.asarray(s).astype(str).reshape(-1)
        truth = np.sign(self.y).astype(int)
        prediction = np.sign(self.y_pred).astype(int)
        active = truth != 0

        cross_forward = _symmetric_rate(
            groups, truth, prediction, "1", "-1", active
        )
        cross_reverse = _symmetric_rate(
            groups, truth, prediction, "-1", "1", active
        )
        pc, dc = _contrast(cross_forward, cross_reverse)

        if "00" in groups[active] and "01" in groups[active]:
            within_1 = _symmetric_rate(
                groups, truth, prediction, "01", "01", active
            )
            within_0 = _symmetric_rate(
                groups, truth, prediction, "00", "00", active
            )
            pw, dw = _contrast(within_1, within_0)
        else:
            pw, dw = 1.0, 0.0
        return CPResult(pc=pc, dc=dc, pw=pw, dw=dw)


def _symmetric_rate(
    groups: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    positive_group: str,
    negative_group: str,
    active: np.ndarray,
) -> tuple[float, float, int]:
    positive = active & (groups == positive_group) & (truth == 1)
    negative = active & (groups == negative_group) & (truth == -1)
    n = int(positive.sum() + negative.sum())
    correct = int(
        (prediction[positive] == 1).sum() + (prediction[negative] == -1).sum()
    )
    rate = correct / n
    return rate, rate * (1 - rate) / n, n


def _contrast(
    first: tuple[float, float, int],
    second: tuple[float, float, int],
) -> tuple[float, float]:
    first_rate, first_variance, first_n = first
    second_rate, second_variance, second_n = second
    gap = first_rate - second_rate
    variance = first_variance + second_variance
    p_value = 1.0 if variance == 0 and gap == 0 else float(
        2 * norm.sf(abs(gap / np.sqrt(variance)))
    )
    pooled = (
        first_variance * first_n**2 + second_variance * second_n**2
    ) / (first_n + second_n)
    effect = 0.0 if pooled == 0 else float(gap / np.sqrt(pooled))
    return p_value, effect
