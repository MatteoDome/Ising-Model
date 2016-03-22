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

def find_equivalent_label(list_of_labels, label):
    while label != list_of_labels[label]:
        label = list_of_labels[label]

    return label

@jit
def find_clusters(N, lattice, links):
    largest_label = -1 
    label = -np.ones([N, N])
    list_of_labels = np.arange(N**2)
    
    for i, j in itertools.product(range(N), range(N)):
        previous_label = label[i, j]
        link_above = links[i, j, 0]
        label_above = label[(i-1)%N, j]
        link_left = links[i, j, 1]
        label_left = label[i, (j-1)%N]

        #   No links so it's a new cluster. Therefore we create a new label
        if not link_above and not link_left:
            largest_label += 1
            label[i, j] = largest_label

        #   One neighbour to the left, existing cluster
        elif link_left and not link_above:
            #   If the neighbour is on the other side of the lattice we create a new label and label that spin too
            if label_left == -1:
                largest_label += 1 
                label[i, (j-1)%N] = largest_label
                label[i, j] = largest_label

            else:
                label[i, j] = find_equivalent_label(list_of_labels, label_left)

        #   One neighbour above, existing cluster
        elif link_above and not link_left:
            #   If there's a neighbour on the other side of the lattice we create a new label and label that spin too
            if label_above == -1:
                largest_label += 1 
                label[(i-1)%N, j] = largest_label
                label[i, j] = largest_label
            
            else:
                label[i, j] = find_equivalent_label(list_of_labels, label_above)

        #   Else neighbours both to the left and above, we link the labels
        else:
            if label_left == -1 and label_above != -1:
                label[i, j] = find_equivalent_label(list_of_labels, label_above)
                label[i, (j-1)%N] = find_equivalent_label(list_of_labels, label_above)
            
            elif label_above == -1 and label_left != -1:
                label[i, j] = find_equivalent_label(list_of_labels, label_left)
                label[(i-1)%N, j] = find_equivalent_label(list_of_labels, label_left)
            
            elif label_above == -1 and label_left == -1:
                largest_label += 1 
                label[i, (j-1)%N] = largest_label
                label[(i-1)%N, j] = largest_label
                label[i, j] = largest_label
            
            else:
                max_label = max(label_left, label_above)
                min_label = min(label_left, label_above)
                list_of_labels[find_equivalent_label(list_of_labels, max_label)] = list_of_labels[find_equivalent_label(list_of_labels, min_label)]
                label[i, j] = min_label

        #   If this site has been visited before and changed its label then we also link the previous label with the new one
        if previous_label != label[i, j] and previous_label != -1:
            list_of_labels[find_equivalent_label(list_of_labels, previous_label)] = find_equivalent_label(list_of_labels, label[i,j])

    return label, list_of_labels

#   THIS CAN BE PARALLEL!!!
@jit
def assign_new_cluster_spins(N, labels, list_of_labels):
    new_lattice = np.zeros([N, N])
    new_spins = np.random.choice([1, -1], size=list_of_labels.size)

    for i, j in itertools.product(range(N), range(N)):
        label = labels[i, j]
        while(label != list_of_labels[label]):
            label = list_of_labels[label]

        new_lattice[i, j] = new_spins[label]

    return new_lattice

def compute_energy(lattice):
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    neighbour_sum = convolve(lattice, k, mode='wrap')
    energy = -0.5*(np.sum(neighbour_sum*lattice))

    return energy

# plt.ion()
# fig = plt.figure()
if __name__ == '__main__':
    #   Simulation parameters
    N = 32
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

    #   Compute number of iterations
    n_iter = int((betaJ_end-betaJ_init)/betaJ_step*n_idle)

    #   Physical quantities to track
    keys = [round(betaJ_init + i*betaJ_step, 2) for i in range(int((betaJ_end-betaJ_init)/betaJ_step)+1)]
    magnetization = dict((betaJ, []) for betaJ in keys)
    energy = dict((betaJ, []) for betaJ in keys)
    susceptibility = dict((betaJ, []) for betaJ in keys)
    binder_cumulant = dict((betaJ, []) for betaJ in keys)
    cv = dict((betaJ, []) for betaJ in keys)

    for i in range(n_iter):
        links = find_links(N, lattice, betaJ)
        cluster_labels, list_of_labels = find_clusters(N, lattice, links)
        lattice = assign_new_cluster_spins(N, cluster_labels, list_of_labels)

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