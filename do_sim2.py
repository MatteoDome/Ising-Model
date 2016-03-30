import pickle
import numpy as np
import simulate_metropolis
import matplotlib.pyplot as plt

N=64
betaJ_init = 0
betaJ_end = 1
betaJ_step = 0.001
n_idle = 5000
second_neighbours = True

magnetization, susceptibility, _, cv = simulate_metropolis  .simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle, second_neighbours)

with open('scaling2/' + str(N) + '_magnetization.dat', 'w') as magnetization_file:
    for point in magnetization:
        magnetization_file.write(str(point[0]) + ' ' + str(point[1]) + '\n')

with open('scaling2/' + str(N) + '_susceptibility.dat', 'w') as susceptibility_file:
    for point in susceptibility:
        susceptibility_file.write(str(point[0]) + ' ' + str(point[1]) + '\n')

with open('scaling2/' + str(N) + '_cv.dat', 'w') as cv_file:
    for point in cv:
        cv_file.write(str(point[0]) + ' ' + str(point[1])+ '\n')

plt.scatter(*zip(*susceptibility))
plt.show()

