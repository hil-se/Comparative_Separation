from simulation import Simulate
from pdb import set_trace

#define underlying true probabilities
#
r1 = 0.231 #P(C = 1, Y = 1, A = 1)
r2 = 0.044 #P(C = 0, Y = 1, A = 1)
r3 = 0.081 #P(C = 1, Y = 0, A = 1)
r4 = 0.144 #P(C = 0, Y = 0, A = 1)
r5 = 0.171 #P(C = 1, Y = 1, A = 0)
r6 = 0.054 #P(C = 0, Y = 1, A = 0)
r7 = 0.121 #P(C = 1, Y = 0, A = 0)
r8 = 0.154 #P(C = 0, Y = 0, A = 0)

p = [r1, r2, r3, r4, r5, r6, r7, r8]

n = 1000
nc = 2000
r = 10000

sim = Simulate(p, alpha = 0.05)
spr = sim.separation_positive_rate(n)
cpr = sim.comparative_separation_positive_rate(nc)
set_trace()
sprx = sim.simulate_separation(n, r)
cprx = sim.simulate_comparative_separation(nc, r)