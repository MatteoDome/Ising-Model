import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import math
import itertools
from numba import jit
from cluster_finding import canonical_label, link_labels, find_clusters, assign_new_cluster_spins, find_links

def simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle):
    #   Simulation variables
    lattice = np.random.choice([1, -1], size=[N, N])
    betaJ = betaJ_init
    n_iter = int((betaJ_end - betaJ_init) / betaJ_step * n_idle)
    neighbour_list = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    #   Physical quantities to track
    magnetization = { betaJ_init : np.array([])}
    energy = { betaJ_init : np.array([]) }
    lat_sum = { betaJ_init : np.array([]) }
    unsubtracted = { betaJ_init : np.array([]) }

    #   Main cycle
    for i in range(n_iter):
        links = find_links(N, lattice, betaJ)
        cluster_labels, label_list, cluster_count = find_clusters(N, links)
        lattice = assign_new_cluster_spins(N, cluster_labels, label_list)

        #   Compute and store physical quantites
        neighbour_sum = convolve(lattice, neighbour_list, mode='wrap')
        energy[betaJ] = np.append(energy[betaJ], -0.5 * (np.sum(neighbour_sum * lattice)))

        magnetization[betaJ] = np.append(magnetization[betaJ], abs(np.mean(lattice)))
        lat_sum[betaJ] = np.append(lat_sum[betaJ], np.sum(lattice))
        unsubtracted[betaJ] = np.append(unsubtracted[betaJ], np.sum(cluster_count**2)/(N*N))

        if i % n_idle == 0:
            betaJ = betaJ + betaJ_step

            magnetization[betaJ] = np.array([])
            energy[betaJ] = np.array([]) 
            lat_sum[betaJ] =  np.array([])
            unsubtracted[betaJ] =  np.array([])

            print("BetaJ: " + str(betaJ))

    #   Process data
    magnetization = [(betaJ, np.mean(magnetization[betaJ])) for betaJ in magnetization]
    susceptibility = [(betaJ, (np.mean(lat_sum[betaJ]**2)-(np.mean(abs(lat_sum[betaJ]))**2))/N**2) for betaJ in lat_sum]
    # susceptibility = [(betaJ, np.mean(unsubtracted[betaJ])) for betaJ in unsubtracted]
    binder_cumulant = [(betaJ, 1 - np.mean(lat_sum[betaJ]**4) /
                        (3 * np.mean(lat_sum[betaJ]**2)**2)) for betaJ in lat_sum]
    cv = [(betaJ, (betaJ**2 * (np.var(energy[betaJ]))) / N**2) for betaJ in energy]

    return magnetization, susceptibility, binder_cumulant, cv

if __name__ == '__main__':
    #   Default simulation parameters
    N = 32
    betaJ_init = 0.1
    betaJ_end = 0.6
    betaJ_step = 0.01
    n_idle = 100

    magnetization, susceptibility, binder_cumulant, cv = simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle)

    plt.scatter(*zip(*magnetization))
    plt.show()
