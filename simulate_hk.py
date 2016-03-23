import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage import convolve
import math
import itertools
from numba import jit


def compute_energy(lattice, neighbour_list):
    """

    Compute the energy of the lattice. neighbour_list is a matrix indicating which
    neighbour spins should be considered when computing the energy (1 contributes, 0
    does not).

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


@jit
def find_clusters(N, lattice, links):
    largest_label = -1
    cluster_labels = -np.ones([N, N], dtype='int_')
    label_list = np.arange(N**2, dtype='int_')
    ncluster = np.zeros(N**2, dtype = 'int_')
    for i, j in itertools.product(range(N), range(N)):
        previous_label = cluster_labels[i, j]
        link_above = links[i, j, 0]
        label_above = cluster_labels[(i - 1) % N, j]
        link_left = links[i, j, 1]
        label_left = cluster_labels[i, (j - 1) % N]

        #   No links so it's a new cluster. Therefore we create a new label
        if not link_above and not link_left:
            largest_label += 1
            cluster_labels[i, j] = largest_label

        #   One neighbour to the left, existing cluster
        elif link_left and not link_above:
            # If the neighbour is on the other side of the lattice we create a
            # new label and label that spin too
            if label_left == -1:
                largest_label += 1
                cluster_labels[i, (j - 1) % N] = largest_label
                cluster_labels[i, j] = largest_label

            else:
                cluster_labels[i, j] = canonical_label(label_list, label_left)

        #   One neighbour above, existing cluster
        elif link_above and not link_left:
            # If there's a neighbour on the other side of the lattice we create
            # a new label and label that spin too
            if label_above == -1:
                largest_label += 1
                cluster_labels[(i - 1) % N, j] = largest_label
                cluster_labels[i, j] = largest_label

            else:
                cluster_labels[i, j] = canonical_label(label_list, label_above)

        #   Else neighbours both to the left and above, we link the labels
        else:
            if label_left == -1 and label_above != -1:
                cluster_labels[i, j] = canonical_label(label_list, label_above)
                cluster_labels[i, (j - 1) % N] = canonical_label(label_list, label_above)

            elif label_above == -1 and label_left != -1:
                cluster_labels[i, j] = canonical_label(label_list, label_left)
                cluster_labels[(i - 1) % N, j] = canonical_label(label_list, label_left)

            #   kinda dumb because this only happens for (0, 0)
            elif label_above == -1 and label_left == -1:
                largest_label += 1
                cluster_labels[i, (j - 1) % N] = largest_label
                cluster_labels[(i - 1) % N, j] = largest_label
                cluster_labels[i, j] = largest_label

            else:
                max_label = max(label_left, label_above)
                min_label = min(label_left, label_above)
                label_list[canonical_label(label_list, max_label)] = label_list[canonical_label(label_list, min_label)]
                cluster_labels[i, j] = min_label
            
        # If this site has been visited before and changed its label then we
        # also link the previous label with the new one
        if previous_label != cluster_labels[i, j] and previous_label != -1:
            label_list[canonical_label(label_list, previous_label)] = canonical_label(label_list, cluster_labels[i, j])

        #if we re-label the boundary sites, we have to modify the cluster numbers accordingly
        ncluster[cluster_labels[i,j]] +=1

    #   Keep only labels that were used
    label_list = label_list[0:largest_label + 1]

    return cluster_labels, label_list, ncluster

def n_rearrange(N, ncluster, label_list):
    for label in label_list:
        if canonical_label(label_list, label) != label:
            ncluster[canonical_label(label_list, label)] += ncluster[label]
            ncluster[label] = 0
    ncluster = ncluster[ncluster>0]
    return ncluster

def assign_new_cluster_spins(N, cluster_labels, label_list):
    new_spins = np.random.choice([1, -1], size=label_list.size)
    new_lattice = np.array(
        [[new_spins[label_list[cluster_labels[i, j]]] for j in range(N)] for i in range(N)])

    return new_lattice

# plt.ion()
# fig = plt.figure()
if __name__ == '__main__':
    #   Simulation parameters
    N = 100
    betaJ_init = 0.35
    betaJ_end = 0.8
    betaJ_step = 0.01
    n_idle = 20
    neighbour_list = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    #   Simulation variables
    lattice = np.random.choice([1, -1], size=[N, N])
    betaJ = betaJ_init
    largest_label = 0
    label = np.zeros([N, N])
    links = np.zeros([N, N, 2])
    #   Compute number of iterations
    n_iter = int((betaJ_end - betaJ_init) / betaJ_step * n_idle)

    #   Physical quantities to track
    keys = [round(betaJ_init + i * betaJ_step, 2)
            for i in range(int((betaJ_end - betaJ_init) / betaJ_step) + 1)]
    magnetization = dict((betaJ, []) for betaJ in keys)
    energy = dict((betaJ, []) for betaJ in keys)
    susceptibility1 = dict((betaJ, []) for betaJ in keys)
    susceptibility2 = dict((betaJ, []) for betaJ in keys)
    binder_cumulant = dict((betaJ, []) for betaJ in keys)
    unsubtr = dict((betaJ, []) for betaJ in keys)
    cv = dict((betaJ, []) for betaJ in keys)

    for i in range(n_iter):
        links = find_links(N, lattice, betaJ)
        cluster_labels, label_list, ncluster = find_clusters(N, lattice, links)
        ncluster = n_rearrange(N, ncluster, label_list)

        #   Reprocess label list
        label_list = np.array([canonical_label(label_list, label)
                               for label in label_list])
        lattice = assign_new_cluster_spins(N, cluster_labels, label_list)

        #   Store physical quantites
        magnetization[betaJ].append(abs(np.mean(lattice)))
        energy[betaJ].append(compute_energy(lattice, neighbour_list))
        susceptibility1[betaJ].append(np.sum(lattice)**2)
        susceptibility2[betaJ].append(abs(np.sum(lattice)))
        unsubtr[betaJ].append(np.sum(ncluster*ncluster)/(N*N))

        if i % n_idle == 0:
            betaJ = round(betaJ + 0.01, 2)
            print(betaJ)

        # fig.clf()
        # ax = fig.add_subplot(111)
        # ax.matshow(lattice)
        # plt.draw()

        # print(i)

    # magnetization_av = [(betaJ, np.mean(magnetization[betaJ])) for betaJ in magnetization]
    # plt.scatter(*zip(*magnetization_av))
    # plt.show()

    susceptibility_av = [(betaJ, np.mean(susceptibility1[betaJ])-(np.mean(susceptibility2[betaJ])**2)) for betaJ in (susceptibility1 and susceptibility2)]
    plt.scatter(*zip(*susceptibility_av))
    plt.show()
    # Cv = np.var(E)*betaJ/(N*N)
    # print(Cv)
