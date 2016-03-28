import pickle
import numpy as np
import simulate_hk
import matplotlib.pyplot as plt

N=256
betaJ_init = 0.300
betaJ_end = 0.600
betaJ_step = 0.001
n_idle = 2000

magnetization, susceptibility, _, cv = simulate_hk.simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle)

with open('scaling/' + str(N) + '_magnetization.dat', 'w') as magnetization_file:
    for point in magnetization:
        magnetization_file.write(str(point[0]) + ' ' + str(point[1]) + '\n')

with open('scaling/' + str(N) + '_susceptibility.dat', 'w') as susceptibility_file:
    for point in susceptibility:
        susceptibility_file.write(str(point[0]) + ' ' + str(point[1]) + '\n')

with open('scaling/' + str(N) + '_cv.dat', 'w') as cv_file:
    for point in cv:
        cv_file.write(str(point[0]) + ' ' + str(point[1])+ '\n')

plt.scatter(*zip(*susceptibility))
plt.show()

