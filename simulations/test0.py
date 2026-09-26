from simulation import Simulate
from pdb import set_trace

#define underlying true probabilities

r1 = 0.22 #P(C = 1, Y = 1, A = 1)
r2 = 0.055 #P(C = 0, Y = 1, A = 1)
r3 = 0.09 #P(C = 1, Y = 0, A = 1)
r4 = 0.135 #P(C = 0, Y = 0, A = 1)
r5 = 0.18 #P(C = 1, Y = 1, A = 0)
r6 = 0.045 #P(C = 0, Y = 1, A = 0)
r7 = 0.11 #P(C = 1, Y = 0, A = 0)
r8 = 0.165 #P(C = 0, Y = 0, A = 0)

p = [r1, r2, r3, r4, r5, r6, r7, r8]

# Init and calculate metrics
sim = Simulate(p, alpha = 0.05)

n = 1000
np = 2000
r = 10000
print("n = %d, np = %d, r = %d"%(n, np, r))


# Calculate sampled variances of TPR(A=1) and TPR(1,0)
var_tpr1 = sim.TPR1 * (1 - sim.TPR1) / (n * (p[0] + p[1]))
var_tpr10 = sim.TPR10 * (1 - sim.TPR10) / ((p[0] + p[1]) * (p[7] + p[6]) * 2 * np)
print("TPR(A=1): %.5f" % sim.TPR1)
print("Expected variance of TPR(A=1): %.5f" % var_tpr1)
print("TPR(1,0): %.5f" % sim.TPR10)
print("Expected variance of TPR(1,0): %.5f" % var_tpr10)
# Simulate sampled variances of TPR(A=1) and TPR(1,0) with r = 10,000 repeats
sim.simulate_tpr1(n, r)
sim.simulate_tpr10(np, r)

# Calculate positive rate
spr = sim.separation_positive_rate(n)
cpr = sim.comparative_separation_positive_rate(np)

# Simulate positive rate
sprx = sim.simulate_separation(n, r)
cprx = sim.simulate_comparative_separation(np, r)

n = 2000
np = 4000
r = 10000
print("n = %d, np = %d, r = %d"%(n, np, r))

# Calculate sampled variances of TPR(A=1) and TPR(1,0)
var_tpr1 = sim.TPR1 * (1 - sim.TPR1) / (n * (p[0] + p[1]))
var_tpr10 = sim.TPR10 * (1 - sim.TPR10) / ((p[0] + p[1]) * (p[7] + p[6]) * 2 * np)
print("TPR(A=1): %.5f" % sim.TPR1)
print("Expected variance of TPR(A=1): %.5f" % var_tpr1)
print("TPR(1,0): %.5f" % sim.TPR10)
print("Expected variance of TPR(1,0): %.5f" % var_tpr10)
# Simulate sampled variances of TPR(A=1) and TPR(1,0) with r = 10,000 repeats
sim.simulate_tpr1(n, r)
sim.simulate_tpr10(np, r)

# Calculate positive rate
spr = sim.separation_positive_rate(n)
cpr = sim.comparative_separation_positive_rate(np)

# Simulate positive rate
sprx = sim.simulate_separation(n, r)
cprx = sim.simulate_comparative_separation(np, r)
