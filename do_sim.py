import numpy as np
import simulate_hk
import matplotlib.pyplot as plt

N=4
betaJ_init = 0.1
betaJ_end = 0.6
betaJ_step = 0.01
n_idle = 10000

magnetization, susceptibility, binder_cumulant, cv = simulate_hk.simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle)

plt.scatter(*zip(*susceptibility))
plt.show()

