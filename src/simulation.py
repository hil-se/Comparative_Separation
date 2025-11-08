import numpy as np
from collections import Counter
from scipy.stats import norm

class Simulate:
    def __init__(self, p, alpha = 0.05):
        self.p = p
        self.alpha = alpha
        self.a = ["111", "011", "101", "001", "110", "010", "100", "000"]
        self.true_metrics()

    def stats(self, x1, x2):
        mu = x1 / (x1+x2)
        var = mu*(1-mu) / (x1+x2)
        return mu, var
    
    def stats_comp(self, xt1, x1, xt2, x2):
        mu = (xt1+xt2) / (x1+x2)
        var = mu*(1-mu) / (x1+x2)
        return mu, var
    
    # Test separation
    def separation(self, x):
        mut1, vart1 = self.stats(x["111"], x["011"])
        mut0, vart0 = self.stats(x["110"], x["010"])
        zt = (mut1-mut0)/np.sqrt(vart1+vart0)
        pt = norm.sf(np.abs(zt))*2
        muf1, varf1 = self.stats(x["101"], x["001"])
        muf0, varf0 = self.stats(x["100"], x["000"])
        zf = (muf1 - muf0) / np.sqrt(varf1 + varf0)
        pf = norm.sf(np.abs(zf))*2
        return [pt, pf]
    
    # Test comparative separation
    def comparative_separation(self, x):
    
        mut11, vart11 = self.stats_comp(x["1111"], x["1111"]+x["0111"]+x["x111"], x["0011"], x["0011"]+x["1011"]+x["x011"])
        mut00, vart00 = self.stats_comp(x["1100"], x["1100"]+x["0100"]+x["x100"], x["0000"], x["0000"]+x["1000"]+x["x000"])
        mut10, vart10 = self.stats_comp(x["1110"], x["1110"]+x["0110"]+x["x110"], x["0001"], x["0001"]+x["1001"]+x["x001"])
        mut01, vart01 = self.stats_comp(x["1101"], x["1101"]+x["0101"]+x["x101"], x["0010"], x["0010"]+x["1010"]+x["x010"])
        zc = (mut10 - mut01) / np.sqrt(vart10 + vart01)
        zw = (mut11 - mut00) / np.sqrt(vart11 + vart00)
        pc = norm.sf(np.abs(zc)) * 2
        pw = norm.sf(np.abs(zw)) * 2
        return [pc, pw]
    
    def type2error(self, w, v, nw, nv):
        var = w * (1 - w) / nw + v * (1 - v) / nv
        mu = w - v
        z = mu / np.sqrt(var)
        p = norm.sf(-1.96 - z) - norm.sf(1.96 - z)
        return p

    def true_metrics(self):
        self.TPR1 = self.p[0] / (self.p[0] + self.p[1])
        self.TPR0 = self.p[4] / (self.p[4] + self.p[5])
        self.FPR1 = self.p[2] / (self.p[2] + self.p[3])
        self.FPR0 = self.p[6] / (self.p[6] + self.p[7])
        self.TPR10 = self.TPR1 * (1 - self.FPR0)
        self.TPR01 = self.TPR0 * (1 - self.FPR1)
        self.TPR11 = self.TPR1 * (1 - self.FPR1)
        self.TPR00 = self.TPR0 * (1 - self.FPR0)
        print("TPRd: %.3f, FPRd: %.3f" % (self.TPR1 - self.TPR0, self.FPR1 - self.FPR0))
        print("TPRc: %.3f, TPRw: %.3f" % (self.TPR10 - self.TPR01, self.TPR11 - self.TPR00))
        print("Calculated Type I error rate: %.4f" %(1-(1-self.alpha)**2))
    
    def separation_positive_rate(self, n = 1000):
        # Estimate type2error for separation
        ndt = self.type2error(self.TPR1, self.TPR0, (self.p[0] + self.p[1]) * n, (self.p[4] + self.p[5]) * n)
        ndf = self.type2error(self.FPR1, self.FPR0, (self.p[2] + self.p[3]) * n, (self.p[6] + self.p[7]) * n)
        spr = 1 - ndt * ndf
        print("ndt: %f, ndf: %f" % (ndt, ndf))
        print("estimated separation positive rate: %.4f" % spr)
        return spr
    
    def comparative_separation_positive_rate(self, nc = 2000):        
        # Estimate type2error for comparative separation
        ndc = self.type2error(self.TPR10, self.TPR01, (self.p[0] + self.p[1]) * (self.p[7] + self.p[6]) * 2 * nc,
                         (self.p[4] + self.p[5]) * (self.p[3] + self.p[2]) * 2 * nc)
        ndw = self.type2error(self.TPR11, self.TPR00, (self.p[0] + self.p[1]) * (self.p[2] + self.p[3]) * 2 * nc,
                         (self.p[4] + self.p[5]) * (self.p[6] + self.p[7]) * 2 * nc)
        cpr = 1 - ndc * ndw
        print("ndc: %f, ndw: %f" % (ndc, ndw))
        print("estimated comparative separation positive rate: %.4f" % cpr)
        return cpr
    
    def simulate_separation(self, n = 1000, r = 10000):
        violate = 0
        for i in range(r):
            selectedr = np.random.choice(self.a, size=n, p=self.p)
            x = Counter(selectedr)
            ps = self.separation(x)
            if min((ps)) < self.alpha:
                violate += 1
        sprx = violate / r
        print("Simulated separation positive rate: %.4f" % sprx)
        return sprx
    
    def simulate_comparative_separation(self, nc = 2000, r = 10000):
        violate = 0
        mus = []
        vars = []
        for i in range(r):
            x = []
            c1 = np.random.choice(self.a, size=nc, p=self.p)
            c2 = np.random.choice(self.a, size=nc, p=self.p)
            for j in range(nc):
                if c1[j][1] == c2[j][1]:
                    continue
                if c1[j][0] == c2[j][0]:
                    cij = "x"
                elif c1[j][0] > c2[j][0]:
                    cij = "1"
                else:
                    cij = "0"
                aij = c1[j][2] + c2[j][2]
                xij = cij + c1[j][1] + aij
                x.append(xij)
            count = Counter(x)

            ps = self.comparative_separation(count)
            if min((ps)) < self.alpha:
                violate += 1
        cprx = violate / r
        print("Simulated comparative separation positive rate: %.4f" %cprx)
        return cprx










