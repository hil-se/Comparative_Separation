import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


class DensityKernel():
    def __init__(self, kernel='gaussian', bandwidth=0.2):
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    def density(self, X):
        scaler = StandardScaler()
        X_z = scaler.fit_transform(X)
        self.kde.fit(X_z)
        w = np.exp(self.kde.score_samples(X_z))
        return w


class DensityNeighbor():
    def __init__(self, radius=0.5, distance='euclidean'):
        self.radius = radius
        self.dist = DistanceMetric.get_metric(distance)

    def density(self, X):
        scaler = StandardScaler()
        X_z = scaler.fit_transform(X)
        dists = self.dist.pairwise(X_z)
        return np.sum(dists < self.radius, axis=1)


