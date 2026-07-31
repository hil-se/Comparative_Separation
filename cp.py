import math

import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import t, norm, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from pdb import set_trace


class CP:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = np.array(y)
        self.y_pred = np.array(y_pred)

    def stats_comp(self, xt1, x1, xt2, x2):
        mu = (xt1 + xt2) / (x1 + x2)
        var = mu * (1 - mu) / (x1 + x2)
        var_pop = mu * (1 - mu)
        return mu, var, var_pop

    # Test comparative separation
    def comparative_separation(self, s):
        # s in {"-1", "1", "00", "01"} where -1 means sa<sb, 1 means sa>sb, 00 means sa=sb=0, 01 means sa=sb=1
        count = []
        for i in range(len(self.y)):
            if self.y[i] < 0:
                y = -1
            elif self.y[i] > 0:
                y = 1
            else:
                continue
            if self.y_pred[i]>0:
                pred = 1
            elif self.y_pred[i]<0:
                pred = -1
            else:
                pred = 0
            count.append((s, y, pred))
        x = Counter(count)

        mut10, vart10, varpop10 = self.stats_comp(x[("1", 1, 1)], x[("1", 1, 1)] + x[("1", 1, -1)] + x[("1", 1, 0)], x[("-1", -1, -1)],
                                        x[("-1", -1, -1)] + x[("-1", -1, 1)] + x[("-1", -1, 0)])
        mut01, vart01, varpop01 = self.stats_comp(x[("-1", 1, 1)], x[("-1", 1, 1)] + x[("-1", 1, -1)] + x[("-1", 1, 0)], x[("1", -1, -1)],
                                        x[("1", -1, -1)] + x[("1", -1, 1)] + x[("1", -1, 0)])
        zc = (mut10 - mut01) / np.sqrt(vart10 + vart01)
        pc = norm.sf(np.abs(zc)) * 2
        dc = (mut10 - mut01) / np.sqrt(varpop10 + varpop01)

        if len(set(s))==4:
            mut11, vart11, varpop11 = self.stats_comp(x[("01", 1, 1)], x[("01", 1, 1)] + x[("01", 1, -1)] + x[("01", 1, 0)], x[("01", -1, -1)],
                                            x[("01", -1, -1)] + x[("01", -1, 1)] + x[("01", -1, 0)])
            mut00, vart00, varpop00 = self.stats_comp(x[("00", 1, 1)], x[("00", 1, 1)] + x[("00", 1, -1)] + x[("00", 1, 0)], x[("00", -1, -1)],
                                            x[("00", -1, -1)] + x[("00", -1, 1)] + x[("00", -1, 0)])
            zw = (mut11 - mut00) / np.sqrt(vart11 + vart00)
            pw = norm.sf(np.abs(zw)) * 2
            dw = (mut11 - mut00) / np.sqrt(varpop11 + varpop00)
        else:
            pw=1.0
            dw=0.0
        return [pc, dc, pw, dw]

    def type2error(self, w, v, nw, nv):
        var = w * (1 - w) / nw + v * (1 - v) / nv
        mu = w - v
        z = mu / np.sqrt(var)
        p = norm.sf(-1.96 - z) - norm.sf(1.96 - z)
        return p

