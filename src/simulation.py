import random
import numpy as np
from collections import Counter
from scipy.stats import norm
from pdb import set_trace


def cal(x1, x2):
    mu = x1 / (x1+x2)
    var = mu*(1-mu) / (x1+x2)
    return mu, var

def cal_comp(xt1, x1, xt2, x2):
    mu = (xt1+xt2) / (x1+x2)
    var = mu*(1-mu) / (x1+x2)
    return mu, var

# Test separation
def separation(x):
    mut1, vart1 = cal(x["111"], x["011"])
    mut0, vart0 = cal(x["110"], x["010"])
    zt = (mut1-mut0)/np.sqrt(vart1+vart0)
    pt = norm.sf(np.abs(zt))*2
    muf1, varf1 = cal(x["101"], x["001"])
    muf0, varf0 = cal(x["100"], x["000"])
    zf = (muf1 - muf0) / np.sqrt(varf1 + varf0)
    pf = norm.sf(np.abs(zf))*2
    return [pt, pf]

# Test comparative separation
def comparative_separation(x):

    mut11, vart11 = cal_comp(x["1111"], x["1111"]+x["0111"]+x["x111"], x["0011"], x["0011"]+x["1011"]+x["x011"])
    mut00, vart00 = cal_comp(x["1100"], x["1100"]+x["0100"]+x["x100"], x["0000"], x["0000"]+x["1000"]+x["x000"])
    mut10, vart10 = cal_comp(x["1110"], x["1110"]+x["0110"]+x["x110"], x["0001"], x["0001"]+x["1001"]+x["x001"])
    mut01, vart01 = cal_comp(x["1101"], x["1101"]+x["0101"]+x["x101"], x["0010"], x["0010"]+x["1010"]+x["x010"])
    zc = (mut10 - mut01) / np.sqrt(vart10 + vart01)
    zw = (mut11 - mut00) / np.sqrt(vart11 + vart00)
    pc = norm.sf(np.abs(zc)) * 2
    pw = norm.sf(np.abs(zw)) * 2
    # zs.append((mut11 - mut00) / np.sqrt(vart11 + vart00))
    # zs.append((mut11 - mut10) / np.sqrt(vart11 + vart10))
    # zs.append((mut11 - mut01) / np.sqrt(vart11 + vart01))
    # zs.append((mut00 - mut10) / np.sqrt(vart00 + vart10))
    # zs.append((mut00 - mut01) / np.sqrt(vart00 + vart01))
    # zs.append((mut10 - mut01) / np.sqrt(vart10 + vart01))
    # ps = []
    # for z in zs:
    #     ps.append(norm.sf(np.abs(z))*2)
    # return mut10, vart10
    return [pc, pw]

#define probability rates
#
# r1 = 0.20 #P(C = 1, Y = 1, A = 1)
# r2 = 0.05 #P(C = 0, Y = 1, A = 1)
# r3 = 0.05 #P(C = 1, Y = 0, A = 1)
# r4 = 0.20 #P(C = 0, Y = 0, A = 1)
# r5 = 0.20 #P(C = 1, Y = 1, A = 0)
# r6 = 0.05 #P(C = 0, Y = 1, A = 0)
# r7 = 0.05 #P(C = 1, Y = 0, A = 0)
# r8 = 0.20 #P(C = 0, Y = 0, A = 0)

# r1 = 0.20 #P(C = 1, Y = 1, A = 1)
# r2 = 0.05 #P(C = 0, Y = 1, A = 1)
# r3 = 0.15 #P(C = 1, Y = 0, A = 1)
# r4 = 0.10 #P(C = 0, Y = 0, A = 1)
# r5 = 0.20 #P(C = 1, Y = 1, A = 0)
# r6 = 0.05 #P(C = 0, Y = 1, A = 0)
# r7 = 0.12 #P(C = 1, Y = 0, A = 0)
# r8 = 0.13 #P(C = 0, Y = 0, A = 0)

# r1 = 0.12 #P(C = 1, Y = 1, A = 1)
# r2 = 0.13 #P(C = 0, Y = 1, A = 1)
# r3 = 0.15 #P(C = 1, Y = 0, A = 1)
# r4 = 0.10 #P(C = 0, Y = 0, A = 1)
# r5 = 0.15 #P(C = 1, Y = 1, A = 0)
# r6 = 0.10 #P(C = 0, Y = 1, A = 0)
# r7 = 0.12 #P(C = 1, Y = 0, A = 0)
# r8 = 0.13 #P(C = 0, Y = 0, A = 0)

r1 = 0.22 #P(C = 1, Y = 1, A = 1)
r2 = 0.03 #P(C = 0, Y = 1, A = 1)
r3 = 0.05 #P(C = 1, Y = 0, A = 1)
r4 = 0.20 #P(C = 0, Y = 0, A = 1)
r5 = 0.23 #P(C = 1, Y = 1, A = 0)
r6 = 0.02 #P(C = 0, Y = 1, A = 0)
r7 = 0.02 #P(C = 1, Y = 0, A = 0)
r8 = 0.23 #P(C = 0, Y = 0, A = 0)

TPR1 = r1/(r1+r2)
TPR0 = r5/(r5+r6)
FPR1 = r3/(r3+r4)
FPR0 = r7/(r7+r8)
print("TPRd: %.2f, FPRd: %.2f" %(TPR1-TPR0, FPR1-FPR0))


print("TPRc: %.2f, TPRw: %.2f" %(TPR1*(1-FPR0)-TPR0*(1-FPR1), TPR1*(1-FPR1)-TPR0*(1-FPR0)))
alpha = 0.05

a = ["111", "011", "101", "001", "110", "010", "100", "000"]
n = 1000
r = 1000
nc = 2*n

# Estimate type 2 error
vart = TPR1*(1-TPR1) / (r1+r2) + TPR0*(1-TPR0) / (r5+r6)
varf = FPR1*(1-FPR1) / (r3+r4) + FPR0*(1-FPR0) / (r7+r8)
mmt = (TPR1-TPR0) / np.sqrt(vart/n)
ndt = norm.sf(-1.96 - mmt)-norm.sf(1.96-mmt)
mmf = (FPR1-FPR0) / np.sqrt(varf/n)
ndf = norm.sf(-1.96 - mmf)-norm.sf(1.96-mmf)
print("ndt: %f, ndf: %f" %(ndt,ndf))
print("estimated positive rate: %f" %(1-ndt*ndf))

TPR10 = TPR1*(1-FPR0)
TPR01 = TPR0*(1-FPR1)
TPR11 = TPR1*(1-FPR1)
TPR00 = TPR0*(1-FPR0)
varc = TPR10*(1-TPR10) / ((r1+r2)*(r8+r7)*2) + TPR01*(1-TPR01) / ((r5+r6)*(r4+r3)*2)
varw = TPR11*(1-TPR11) / ((r1+r2)*(r3+r4)*2) + TPR00*(1-TPR00) / ((r5+r6)*(r7+r8)*2)
mmc = (TPR10-TPR01) / np.sqrt(varc/(nc))
ndc = norm.sf(-1.96 - mmc)-norm.sf(1.96-mmc)
mmw = (TPR11-TPR00) / np.sqrt(varw/(nc))
ndw = norm.sf(-1.96 - mmw)-norm.sf(1.96-mmw)
print("ndc: %f, ndw: %f" %(ndc,ndw))
print("estimated positive rate: %f" %(1-ndc*ndw))
# print("TPR10: %f, var: %f" %(TPR10, TPR10*(1-TPR10) / ((r1+r2)*(r8+r7)*2*nc)))
set_trace()

violate = 0
for i in range(r):
    selectedr = np.random.choice(a, size = n, p = [r1, r2, r3, r4, r5, r6, r7, r8])
    x = Counter(selectedr)
    ps = separation(x)
    if min((ps)) < alpha:
        violate += 1

print(violate/r)

violate = 0
mus = []
vars = []
for i in range(r):
    x = []
    c1 = np.random.choice(a, size=nc, p=[r1, r2, r3, r4, r5, r6, r7, r8])
    c2 = np.random.choice(a, size=nc, p=[r1, r2, r3, r4, r5, r6, r7, r8])
    for j in range(nc):
        if c1[j][1]==c2[j][1]:
            continue
        if c1[j][0] == c2[j][0]:
            cij = "x"
        elif c1[j][0] > c2[j][0]:
            cij = "1"
        else:
            cij = "0"
        aij = c1[j][2]+c2[j][2]
        xij = cij + c1[j][1]+aij
        x.append(xij)
    count = Counter(x)

    ps = comparative_separation(count)
    if min((ps)) < alpha:
        violate += 1
print(violate / r)
#     mut10, vart10 = comparative_separation(count)
#     mus.append(mut10)
#     vars.append(vart10)
# mu = np.mean(mus)
# var1 = np.var(mus)
# var2 = np.mean(vars)
# print("mu: %f, var1: %f, var2: %f" %(mu, var1, var2))


