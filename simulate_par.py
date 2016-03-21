import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage import convolve
import math
import itertools
from numba import jit
import time

def find_links(N, lattice, betaJ):
    prob = np.exp(-2*betaJ)
    links = np.zeros([N, N, 2])

    #   Set links to 1 if they match 
    links[lattice == np.roll(lattice, 1, 0), 0] = 1
    links[lattice == np.roll(lattice, 1, 1), 1] = 1

    #   Keep links with with some probability
    random_matrix = np.random.uniform(0, 1, size = [N, N, 2])
    links[(random_matrix < prob) & (links == 1)] = 0

    return links

@jit
def find_clusters(N, lattice, links):
    largest_label = 0 
    label = -np.ones([N, N])
    label[0,0] = 1
    for i, j in itertools.product(range(N), range(N)):
        if links[i,j,0] == 1 and links[i,j,1] != 1 and label[(i-1)%N, j] != -1:                        #can be done in one if
            label[i,j] = label[(i-1)%N, j]
        elif links[i,j,1] == 1 and links[i,j,0] != 1  and label[i, (j-1)%N] != -1:
            label[i,j] = label[i, (j-1)%N]
        elif all(links[i,j,:]) == 1 and label[(i-1)%N, j] != -1 and label[i, (j-1)%N] != -1:
            label[i,j] = label[(i-1)%N, j]
        # else:
        elif all(links[i,j,:]) != 1 or (links[i,j,1] == 1 and links[i,j,0] != 1  and label[i, (j-1)%N] != -1) or (links[i,j,0] == 1 and links[i,j,1] != 1 and label[(i-1)%N, j] != -1)
            largest_label += 1
            label[i,j] = largest_label
    # print("----")
    # print(largest_label)
    for i, j in itertools.product(range(N), range(N)):
        
        if links[i,j,0] ==1 and label[i,j] != label[(i-1)%N, j]:
            label[i, j] = min(label[i,j], label[(i-1)%N, j])
            label[(i-1)%N, j] = label[i,j]

        if links[i,j,1] ==1 and label[i,j] != label[i, (j-1)%N]:
            label[i, j] = min(label[i,j], label[i, (j-1)%N])
            label[i, (j-1)%N] = label[i,j]
       
        if links[(i+1)%N,j,0] ==1 and label[i,j] != label[(i+1)%N, j]:
            label[i, j] = min(label[i,j], label[(i+1)%N, j])
            label[(i+1)%N, j] = label[i,j]

        if links[i,(j+1)%N,1] ==1 and label[i,j] != label[i, (j+1)%N]:
            label[i, j] = min(label[i,j], label[i, (j+1)%N])
            label[i, (j+1)%N] = label[i,j]
    
    return label, largest_label

def assign_new_cluster_spins(N, label, largest_label):
    up_lattice = np.zeros([N, N])
    new_spins = np.random.choice([1, -1], size = largest_label + 1)
    up_lattice = numpy.array( [ [ new_spins[i, j] for i in range(N) for j in range(N) ] ] )

    return up_lattice

def compute_energy(lattice):
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    neighbour_sum = convolve(lattice, k, mode='wrap')
    energy = -0.5*(np.sum(neighbour_sum*lattice))

    return energy

#   Compute number of iterations
n_iter = int((betaJ_end-betaJ_init)/betaJ_step*n_idle)

#   Physical quantities to track
keys = [round(betaJ_init + i*betaJ_step, 2) for i in range(int((betaJ_end-betaJ_init)/betaJ_step)+1)]
magnetization = dict((betaJ, []) for betaJ in keys)
energy = dict((betaJ, []) for betaJ in keys)
susceptibility = dict((betaJ, []) for betaJ in keys)
binder_cumulant = dict((betaJ, []) for betaJ in keys)
cv = dict((betaJ, []) for betaJ in keys)

# plt.ion()
# fig = plt.figure()
if __name__ == '__main__':
    #   Simulation parameters
    N = 128
    betaJ_init = 0.01
    betaJ_end = 1
    betaJ_step = 0.01
    n_idle = 100

    #   Simulation variables
    lattice = np.random.choice([1, -1], size = [N, N])
    betaJ = betaJ_init
    largest_label = 0
    label = np.zeros([N, N])
    links= np.zeros([N, N, 2])

    
    for i in range(n_iter):
        links = find_links(N, lattice, betaJ)
        cluster_labels, largest_label = find_clusters(N, lattice, links)
        lattice = assign_new_cluster_spins(N, lattice, largest_label)

        magnetization[betaJ].append(abs(np.mean(lattice)))
        energy[betaJ].append(compute_energy(lattice))
        susceptibility[betaJ].append(np.mean(lattice)**2)

        if i%n_idle==0:
            betaJ = round(betaJ + 0.01, 2)
            print(betaJ)

        # fig.clf()
        # ax = fig.add_subplot(111)
        # ax.matshow(lattice)
        # plt.draw()

        # print(i)

    magnetization_av =  [(betaJ, np.mean(magnetization[betaJ])) for betaJ in magnetization]
    plt.scatter(*zip(*magnetization_av))
    plt.show()
        # # if i > n_iter_init:
        # #     E[i-n_iter_init] = compute_energy(lattice)
        # # print("-------")
        # print(i)

    # Cv = np.var(E)*betaJ/(N*N)
    # print(Cv)