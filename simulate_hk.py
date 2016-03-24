import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import math
import itertools
from numba import jit
import time

def compute_energy(lattice, neighbour_list):
    """

    Compute the energy of the lattice. neighbour_list is a matrix indicating which
    neighbour spins should be considered when computing the energy (1 contributes, 0
    does not).
   
    Args
        neighbour_list: 3x3 matrix indicating which neighbour interactions to consider
    """

    #   Wrap mode used for periodic boundary conditions
    neighbour_sum = convolve(lattice, neighbour_list, mode='wrap')
    energy = -0.5 * (np.sum(neighbour_sum * lattice))

    return energy


def find_links(N, lattice, betaJ):
    """
    Marks links to the left and right of each spin in the lattice.

    Args
        N: size of lattice (assumed to be NxN)
        lattice: size of lattice (assumed to be NxN)

    Returns
        An NxNx2 tensor with boolen entries (1 if there is a link, 0 if not).
        Entry [i, j, 0] refers to the neighbour above and [i, j, 1]
        to the neighbour on the left.
    """

    prob = np.exp(-2*betaJ)
    links = np.zeros([N, N, 2], 'int_')

    #   Set links to 1 if they match
    links[lattice == np.roll(lattice, 1, 0), 0] = 1
    links[lattice == np.roll(lattice, 1, 1), 1] = 1

    #   Keep links with with some probability
    random_matrix = np.random.uniform(0, 1, size=[N, N, 2])
    links[(random_matrix < prob) & (links == 1)] = 0

    return links


def canonical_label(label_list, label):
    while label != label_list[label]:
        label = label_list[label]

    return label

def link_labels(label_list, labels_to_link):
    labels_to_link = [canonical_label(label_list, label) for label in labels_to_link]
    label_list[max(labels_to_link)] = min(labels_to_link)
    
    return min(labels_to_link)

@jit
def find_clusters(N, lattice, links):
    largest_label = -1
    cluster_labels = -np.ones([N, N], dtype='int_')
    label_list = np.arange(N**2, dtype='int_')
    cluster_count = np.zeros(N**2, dtype = 'int_')

    #   Throw away links in the boundary, these will be dealt with later
    links[:, 0, 1] = 0
    links[0, :, 0] = 0

    for i, j in itertools.product(range(N), range(N)):
        link_above, link_left  = links[i, j, 0], links[i, j, 1]
        label_above, label_left = cluster_labels[(i - 1) % N, j], cluster_labels[i, (j - 1) % N]

        #   No links so it's a new cluster. Therefore we create a new label
        if not link_above and not link_left:
            largest_label += 1
            cluster_labels[i, j] = largest_label

        #   One neighbour to the left
        elif link_left and not link_above:
            cluster_labels[i, j] = canonical_label(label_list, label_left)

        #   One neighbour above
        elif link_above and not link_left:
            cluster_labels[i, j] = canonical_label(label_list, label_above)

        #   Else neighbours both to the left and above
        else:
            cluster_labels[i, j] = link_labels(label_list, [label_left, label_above])

        cluster_count[cluster_labels[i,j]] +=1
            
    # Iterate through boundaries
    for i in range(N):
        if lattice[i, 0] == lattice[i, N - 1]:
            cluster_labels[i, 0] = link_labels(label_list, [cluster_labels[i, 0], cluster_labels[i, N - 1]])
        if lattice[0, i] == lattice[N - 1, i]:
            cluster_labels[0, i] = link_labels(label_list, [cluster_labels[0, i], cluster_labels[N - 1, i]])

    #   Keep only labels that were used
    label_list = label_list[0:largest_label + 1]
    
    #   Reprocess label and cluster count
    for index, label in enumerate(label_list):
        correct_label = canonical_label(label_list, label)
        if correct_label != label:
            cluster_count[correct_label] += cluster_count[label]
            cluster_count[label] = 0
            label_list[index] = correct_label

    return cluster_labels, label_list, cluster_count

def assign_new_cluster_spins(N, cluster_labels, label_list):
    new_spins = np.random.choice([1, -1], size=label_list.size)
    new_lattice = np.array(
        [[new_spins[label_list[cluster_labels[i, j]]] for j in range(N)] for i in range(N)])

    return new_lattice

def simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle):
    #   Simulation variables
    lattice = np.random.choice([1, -1], size=[N, N])
    betaJ = betaJ_init
    largest_label = 0
    label = np.zeros([N, N])
    links = np.zeros([N, N, 2])
    n_iter = int((betaJ_end - betaJ_init) / betaJ_step * n_idle)
    neighbour_list = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    #   betaJ values that will be sweeped
    betaJ_values = [round(betaJ_init + i * betaJ_step, 2)
            for i in range(int((betaJ_end - betaJ_init) / betaJ_step) + 1)]
    
    #   Physical quantities to track
    magnetization = { betaJ : np.array([]) for betaJ in betaJ_values}
    energy = { betaJ : np.array([]) for betaJ in betaJ_values}
    lat_sum = { betaJ : np.array([]) for betaJ in betaJ_values}
    unsubtracted = { betaJ : np.array([]) for betaJ in betaJ_values}

    #   Main cycle
    for i in range(n_iter):
        links = find_links(N, lattice, betaJ)
        cluster_labels, label_list, cluster_count = find_clusters(N, lattice, links)
        lattice = assign_new_cluster_spins(N, cluster_labels, label_list)

        #   Store physical quantites
        magnetization[betaJ] = np.append(magnetization[betaJ], abs(np.mean(lattice)))
        energy[betaJ] = np.append(energy[betaJ], compute_energy(lattice, neighbour_list))
        lat_sum[betaJ] = np.append(lat_sum[betaJ], np.sum(lattice))
        unsubtracted[betaJ] = np.append(unsubtracted[betaJ], np.sum(cluster_count**2)/(N*N))

        if i % n_idle == 0:
            betaJ = round(betaJ + 0.01, 2)
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
    n_idle = 10

    magnetization, susceptibility, binder_cumulant, cv = simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle)

    plt.scatter(*zip(*susceptibility))
    plt.show()