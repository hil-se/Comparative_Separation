import math

import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import t, norm, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from pdb import set_trace


class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = np.array(y)
        self.y_pred = np.array(y_pred)

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def accuracy(self):
        return sklearn.metrics.accuracy_score(self.y, self.y_pred)

    def f1(self):
        return sklearn.metrics.f1_score(self.y, self.y_pred)

    def precision(self):
        return sklearn.metrics.precision_score(self.y, self.y_pred)

    def recall(self):
        return sklearn.metrics.recall_score(self.y, self.y_pred)

    def r2(self):
        return sklearn.metrics.r2_score(self.y, self.y_pred)

    def pearsonr(self):
        return pearsonr(self.y_pred, self.y)

    def spearmanr(self):
        return spearmanr(self.y_pred, self.y)

    def confusion(self, y, y_pred):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(y)):
            if y[i] > 0:
                if y_pred[i] > 0:
                    tp += 1
                else:
                    fn += 1
            else:
                if y_pred[i] > 0:
                    fp += 1
                else:
                    tn += 1
        return tp, fp, tn, fn

    def stats(self, x1, x2):
        n = x1+x2
        if n:
            mu = x1 / n
            var = mu * (1 - mu) / n
            return mu, var, n
        else:
            return 0,0,0

    def stats_comp(self, xt1, x1, xt2, x2):
        n = (x1 + x2)
        if n:
            mu = (xt1 + xt2) / n
            var = mu * (1 - mu) / n
            return mu, var, n
        else:
            return 0,0,0

    # Test separation
    def separation(self, s, stats=False):
        count = []
        for i in range(len(s)):
            count.append(str(int(self.y_pred[i]))+str(int(self.y[i]))+str(int(s[i])))
        x = Counter(count)

        mut1, vart1, nt1 = self.stats(x["111"], x["011"])
        mut0, vart0, nt0 = self.stats(x["110"], x["010"])
        zt = (mut1 - mut0) / np.sqrt(vart1 + vart0)
        pt = norm.sf(np.abs(zt)) * 2
        dt = (mut1 - mut0) / np.sqrt((vart1 * nt1 * nt1 + vart0 * nt0 * nt0) / (nt1 + nt0))
        muf1, varf1, nf1 = self.stats(x["101"], x["001"])
        muf0, varf0, nf0 = self.stats(x["100"], x["000"])
        zf = (muf1 - muf0) / np.sqrt(varf1 + varf0)
        pf = norm.sf(np.abs(zf)) * 2
        df = (muf1 - muf0) / np.sqrt((varf1 * nf1 * nf1 + varf0 * nf0 * nf0) / (nf1 + nf0))
        if stats:
            print(x)
            print("TPR1 = %.2f, TPR0 = %.2f, FPR1 = %.2f, FPR0 = %.2f" %(mut1, mut0, muf1, muf0))
        return pt, dt, pf, df

    # Test comparative separation
    def comparative_separation(self, s0, s1, stats=False):
        count = []
        for i in range(len(self.y)):
            if self.y[i] < 0:
                y = 0
            elif self.y[i] > 0:
                y = 1
            else:
                continue
            if self.y_pred[i]>0:
                pred = "1"
            elif self.y_pred[i]<0:
                pred = "0"
            else:
                pred = "x"
            count.append(pred + str(int(y)) + str(int(s0[i])) + str(int(s1[i])))
        x = Counter(count)

        mut11, vart11, n11 = self.stats_comp(x["1111"], x["1111"] + x["0111"] + x["x111"], x["0011"],
                                        x["0011"] + x["1011"] + x["x011"])
        mut00, vart00, n00 = self.stats_comp(x["1100"], x["1100"] + x["0100"] + x["x100"], x["0000"],
                                        x["0000"] + x["1000"] + x["x000"])
        mut10, vart10, n10 = self.stats_comp(x["1110"], x["1110"] + x["0110"] + x["x110"], x["0001"],
                                        x["0001"] + x["1001"] + x["x001"])
        mut01, vart01, n01 = self.stats_comp(x["1101"], x["1101"] + x["0101"] + x["x101"], x["0010"],
                                        x["0010"] + x["1010"] + x["x010"])
        zc = (mut10 - mut01) / np.sqrt(vart10 + vart01)
        pc = norm.sf(np.abs(zc)) * 2
        dc = (mut10 - mut01) / np.sqrt((vart10 * n10 * n10 + vart01 * n01 * n01) / (n10 + n01))
        if n11>0 and n00>0:
            zw = (mut11 - mut00) / np.sqrt(vart11 + vart00)
            pw = norm.sf(np.abs(zw)) * 2
            dw = (mut11 - mut00) / np.sqrt((vart11 * n11 * n11 + vart00 * n00 * n00) / (n11 + n00))
        else:
            pw = 1.0
            dw = 0.0
        if stats:
            print(x)
            print("TPR11 = %.2f, TPR00 = %.2f, TPR10 = %.2f, TPR01 = %.2f" % (mut11, mut00, mut10, mut01))
        return pc, dc, pw, dw

    def type2error(self, w, v, nw, nv):
        var = w * (1 - w) / nw + v * (1 - v) / nv
        mu = w - v
        z = mu / np.sqrt(var)
        p = norm.sf(-1.96 - z) - norm.sf(1.96 - z)
        return p

    def EOD(self, s):
        # True positive rate (TPR)
        y0 = self.y[s == 0]
        y0_pred = self.y_pred[s == 0]
        y1 = self.y[s == 1]
        y1_pred = self.y_pred[s == 1]

        tp, fp, tn, fn = self.confusion(y0, y0_pred)
        op0 = float(tp) / (tp + fn)
        tp, fp, tn, fn = self.confusion(y1, y1_pred)
        op1 = float(tp) / (tp + fn)
        return op1 - op0

    def AOD(self, s):
        # equal TPR and equal FPR
        y0 = self.y[s == 0]
        y0_pred = self.y_pred[s == 0]
        y1 = self.y[s == 1]
        y1_pred = self.y_pred[s == 1]

        tp, fp, tn, fn = self.confusion(y0, y0_pred)
        od0 = float(tp) / (tp + fn) + float(fp) / (fp + tn)
        tp, fp, tn, fn = self.confusion(y1, y1_pred)
        od1 = float(tp) / (tp + fn) + float(fp) / (fp + tn)
        return (od1 - od0) / 2

    def Isep(self, s):

        joint = pd.DataFrame({'y': self.y, 'y_pred': self.y_pred}, columns=['y', 'y_pred'])
        margin = self.y.reshape(-1, 1)
        model_joint = LogisticRegression().fit(joint, s)
        model_margin = LogisticRegression().fit(margin, s)

        prob_joint = model_joint.predict_proba(joint)
        prob_margin = model_margin.predict_proba(margin)
        Info = 0
        Entropy = 0

        for i in range(len(s)):
            Info = Info + np.log(prob_joint[i][s[i]] / prob_margin[i][s[i]])
            Entropy = Entropy + np.log(prob_margin[i][s[i]])

        MI = Info / (-Entropy)
        return MI

    def Csep_xfit(self, s, groups=None, n_splits=10, seed=0, return_raw=False):
        s = np.asarray(s, dtype=float)
        y = np.asarray(self.y, dtype=float)
        pred = np.asarray(self.y_pred, dtype=float)
        
        groups = np.arange(len(s)).astype(str) if groups is None else np.asarray(groups, dtype=str)
        unique_groups = np.unique(groups)
        k = min(n_splits, len(unique_groups))
        if np.unique(s).size == 1:
            return 0.0

        joint = np.column_stack((y, pred))
        margin = y.reshape(-1, 1)
        ratios = []
        for train_groups, test_groups in KFold(k, shuffle=True, random_state=seed).split(unique_groups):
            train = np.isin(groups, unique_groups[train_groups])
            test = np.isin(groups, unique_groups[test_groups])
            log_densities = []
            for x in (joint, margin):
                model = LinearRegression().fit(x[train], s[train])
                scale = max(float(np.std(s[train] - model.predict(x[train]))), 1e-12)
                log_densities.append(norm.logpdf(s[test], model.predict(x[test]), scale))
            ratios.extend(log_densities[0] - log_densities[1])
        raw = float(np.mean(ratios))
        return raw if return_raw else max(0.0, raw)

    def Csep(self, s):

        joint = pd.DataFrame({'y': self.y, 'y_pred': self.y_pred}, columns=['y', 'y_pred'])
        margin = self.y.reshape(-1, 1)
        model_joint = LinearRegression().fit(joint, s)
        model_margin = LinearRegression().fit(margin, s)

        pred_joint = model_joint.predict(joint)
        pred_margin = model_margin.predict(margin)
        rse_joint = np.std(pred_joint - s)
        rse_margin = np.std(pred_margin - s)

        pdf_joint = norm.pdf(s, pred_joint, rse_joint)
        pdf_margin = norm.pdf(s, pred_margin, rse_margin)

        Info = 0

        for i in range(len(s)):
            Info = Info + math.log(pdf_joint[i] / pdf_margin[i])

        MI = Info / len(s)
        return MI
