import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from numba import jit

def find_links(N, lattice, betaJ, **kwargs):
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
    
    if 'parallel_component' in kwargs:
        parallel_component = kwargs['parallel_component']

        #   Compute coupling constants
        coupling = np.zeros([N, N, 2])
        coupling[:, :, 0]  = np.abs(parallel_component)*np.roll(np.abs(parallel_component), 1, 0)
        coupling[:, :, 1]  = np.abs(parallel_component)*np.roll(np.abs(parallel_component), 1, 1)

        prob = np.exp(-2*coupling*betaJ)
   
    else:
        prob = np.exp(-2*betaJ)
    
    links = np.zeros([N, N, 2], 'int_')

    #   Set links to 1 if they match
    links[lattice == np.roll(lattice, 1, 0), 0] = True
    links[lattice == np.roll(lattice, 1, 1), 1] = True

    #   Keep links with with some probability
    random_matrix = np.random.uniform(0, 1, size=[N, N, 2])
    links[(random_matrix < prob) & (links == 1)] = False

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
def find_clusters(N, links):
    largest_label = -1
    cluster_labels = -np.ones([N, N], dtype='int_')
    label_list = np.arange(N**2, dtype='int_')
    cluster_count = np.zeros(N**2, dtype = 'int_')

    #   Throw away links in the boundary and store them in another array, they will be dealt with later
    upper_boundary_links = np.copy(links[0, :, 0])
    links[0, :, 0] = 0
    
    left_boundary_links = np.copy(links[:, 0, 1])
    links[:, 0, 1] = 0

    for i, j in itertools.product(range(N), range(N)):
        link_above, link_left  = links[i, j, 0], links[i, j, 1]
        label_above, label_left = cluster_labels[(i - 1) % N, j], cluster_labels[i, (j - 1) % N]

        #   No links so it's a new cluster. Therefore we create a new label
        if not link_above and not link_left:
            largest_label += 1
            cluster_labels[i, j] = largest_label

        elif link_left and not link_above:
            cluster_labels[i, j] = canonical_label(label_list, label_left)

        elif link_above and not link_left:
            cluster_labels[i, j] = canonical_label(label_list, label_above)

        else:
            cluster_labels[i, j] = link_labels(label_list, [label_left, label_above])

        cluster_count[cluster_labels[i,j]] +=1
            
    # Now we take care of the boundary links
    links[0, :, 0] = upper_boundary_links
    links[:, 0, 1] = left_boundary_links

    for i in range(N):
        if links[i, 0, 1]:
            cluster_labels[i, 0] = link_labels(label_list, [cluster_labels[i, 0], cluster_labels[i, N - 1]])
        if links[0, i, 0]:
            cluster_labels[0, i] = link_labels(label_list, [cluster_labels[0, i], cluster_labels[N - 1, i]])

    #   Keep only labels that were used
    label_list = label_list[0:largest_label + 1]
    
    #   Reprocess label and count the clusters 
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