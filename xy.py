import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import math
import itertools
from numba import jit

def find_links(N, spins, parallel_component, betaJ):
    #   Compute coupling constants
    coupling = np.zeros([N, N, 2])
    coupling[spins == np.roll(spins, 1, 0), 0]  = (np.abs(parallel_component)*np.abs(np.roll(parallel_component, 1, 0)))[spins == np.roll(spins, 1, 0)]
    coupling[spins == np.roll(spins, 1, 1), 1]  = (np.abs(parallel_component)*np.abs(np.roll(parallel_component, 1, 1)))[spins == np.roll(spins, 1, 1)]
    prob = np.exp(-betaJ * coupling)

    #   Compute links
    links = np.zeros([N, N, 2], 'int_')

    #   Set links to 1 if they match
    links[spins == np.roll(spins, 1, 0), 0] = 1
    links[spins == np.roll(spins, 1, 1), 1] = 1

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
            
    # Restore links in the boundary
    links[0, :, 0] = upper_boundary_links
    links[:, 0, 1] = left_boundary_links

    for i in range(N):
        if links[i, 0, 1]:
            cluster_labels[i, 0] = link_labels(label_list, [cluster_labels[i, 0], cluster_labels[i, N - 1]])
        if links[0, i, 0]:
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

N=3

#   Create random spins and normalize
lattice = np.random.uniform(-1, 1, size = [N, N, 3])
lattice = lattice/np.linalg.norm(lattice, axis=2, keepdims=True)

#   Create random vector and normalize
random_vec = np.random.uniform(-1, 1, size = 3)
random_vec = random_vec/np.linalg.norm(random_vec)

#   Compute parallel and perpendicular components
parallel_component = np.sum(lattice*random_vec, axis=2)

#   Create lattice of spin components along random vector
spins = np.zeros(shape=[N, N]) 
spins[parallel_component < 0] = -1 
spins[parallel_component > 0] = 1

parallel_projection = parallel_component[:, :, None] * random_vec
perpendicular_projection = lattice - parallel_projection

links = find_links(N, spins, parallel_component, 1000)
cluster_labels, label_list, cluster_count = find_clusters(N, links)
new_spins = assign_new_cluster_spins(N, cluster_labels, label_list)

#   Flip parallel part of spins
parallel_projection[spins != new_spins, :] = - parallel_projection[spins != new_spins, :]
lattice = parallel_projection + perpendicular_projection
