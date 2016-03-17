import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage import convolve
import math
import itertools
from numba import jit
import time

#   Simulation parameters
N = 50
T = 1
n_iter = 10000
n_iter_init = 1000
betaJ = 0.01
E= np.zeros([n_iter - n_iter_init])

##Hoshen Kopelman

lattice = np.random.choice([1, -1], size = [N, N])
largest_label = 0
label = np.zeros([N, N])
links= np.zeros([N, N, 2])

def link(N, lattice, betaJ):
    start=time.time()
    prob = np.exp(-2*betaJ)
    links = np.zeros([N, N, 2])

    #   Set links to 1 if they match 
    links[lattice == np.roll(lattice, 1, 0), 0] = 1
    links[lattice == np.roll(lattice, 1, 1), 1] = 1

    #   Keep links with with some probability
    random_matrix = np.random.uniform(0, 1, size = [N, N, 2])
    links[(random_matrix < prob) & (links == 1)] = 0
    print(time.time()-start)
    return links

@jit
def latti_upd(N, lattice, links, betaJ):
    largest_label = 0 
    label = np.zeros([N, N])
    for i, j in itertools.product(range(N), range(N)):
        if links[i,j,0] == 1 and links[i,j,1] != 1:                        #can be done in one if
            label[i,j] = label[(i-1)%N, j]
        elif links[i,j,1] == 1 and links[i,j,0] != 1:
            label[i,j] = label[i, (j-1)%N]
        elif links[i,j,1] == 1 and links[i,j,0] == 1:
            label[i,j] = label[(i-1)%N, j]
        elif all(links[i,j,:]) != 1:
            largest_label += 1
            label[i,j] = largest_label
    # print("----")
    # print(largest_label)
    for i, j in itertools.product(range(N), range(N)):
        if links[i,j,0] ==1 and label[i,j] != label[(i-1)%N, j]:
            label[i, j] = min(label[i,j], label[(i-1)%N, j])
            label[(i-1)%N, j] = min(label[i,j], label[(i-1)%N, j])
        if links[i,j,1] ==1 and label[i,j] != label[i, (j-1)%N]:
            label[i, j] = min(label[i,j], label[i, (j-1)%N])
            label[i, (j-1)%N] = min(label[i,j], label[i, (j-1)%N])
        if links[(i+1)%N,j,0] ==1 and label[i,j] != label[(i+1)%N, j]:
            label[i, j] = min(label[i,j], label[(i+1)%N, j])
            label[(i+1)%N, j] = min(label[i,j], label[(i+1)%N, j])
        if links[i,(j+1)%N,1] ==1 and label[i,j] != label[i, (j+1)%N]:
            label[i, j] = min(label[i,j], label[i, (j+1)%N])
            label[i, (j+1)%N] = min(label[i,j], label[i, (j+1)%N])

    largest_label = largest_label +1
    up_lattice = np.zeros([N, N])
    new_spin = np.random.choice([1, -1], size = largest_label)

    for i, j in itertools.product(range(N), range(N)):
        up_lattice[i, j] = new_spin[(label[i,j])]
    
    return up_lattice

def compute_energy(lattice):
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    neighbour_sum = convolve(lattice, k, mode='wrap')
    energy = -0.5*(np.sum(neighbour_sum*lattice))

    return energy

lattice_sum = np.zeros(100)
x = np.zeros(100)
chi = np.zeros(100)
magnetization = np.zeros(100)
# plt.ion()
# fig = plt.figure()

for i in range (0, n_iter):
    links = link(N, lattice, betaJ)
    lattice = latti_upd(N, lattice, links, betaJ)
    lattice_sum[i%100] = abs(lattice.sum())
    if i%100==0:
        x[int(i/100)] = betaJ
        chi[int(i/100)] = np.mean(lattice_sum*lattice_sum)/(N*N*N*N)
        magnetization[int(i/100)] = np.mean(lattice_sum)/(N)
        betaJ+=0.01
        lattice_sum = np.zeros(100)
        print(betaJ)
    # fig.clf()
    # ax = fig.add_subplot(111)
    # ax.matshow(lattice)
    # plt.draw()

    # print(i)

plt.plot(x,magnetization)
plt.show()
    # # if i > n_iter_init:
    # #     E[i-n_iter_init] = compute_energy(lattice)
    # # print("-------")
    # print(i)

# Cv = np.var(E)*betaJ/(N*N)
# print(Cv)