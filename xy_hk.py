import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from numba import jit
from cluster_finding import canonical_label, link_labels, find_clusters, assign_new_cluster_spins, find_links

def compute_helicity_modulus(N, angles, T):
    a = np.cos(angles - np.roll(angles, 1, axis = 2)) + \
        np.cos(angles - np.roll(angles, -1, axis = 2)) + \
        np.cos(angles - np.roll(angles, 1, axis = 1)) + \
        np.cos(angles - np.roll(angles, -1, axis = 1))

    a = np.sum(a, axis = (1, 2))/2

    b = np.sin(angles - np.roll(angles, 1, axis=2))
    b = np.sum(b, axis = (1, 2))
    
    c = np.sin(angles - np.roll(angles, 1, axis=1))
    c = np.sum(c, axis = (1, 2))
    helicity = 1/(2*N**2)*(np.mean(a) - np.mean(b**2)/T - np.mean(c**2)/T)

    return helicity

def simulate(N, T_init, T_end, T_step, n_idle):
    #   Create random spins and normalize
    # lattice = np.random.uniform(-1, 1, size = [N, N, 2])
    # lattice = lattice/np.linalg.norm(lattice, axis=2, keepdims=True)   
    
    #   Create normalized lattice
    lattice = np.ones(shape = [N, N, 2])/(np.sqrt(2)) 
    T = T_init
    n_iter = int((T_end - T_init) / T_step * n_idle)
    neighbour_list = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    #   Physical quantities to track
    helicity_modulus = []
    angles = np.zeros(shape = [n_idle, N, N])
    
    #   Main cycle
    for i in range(n_iter):
        #   Store angles of the spins to later calculate helicity modulus
        angles[int(i%n_idle), :, :] = np.arcsin(lattice[:, :, 1])

        #   Generate random vector and normalize
        random_vec = np.random.uniform(-1, 1, size = 2)
        random_vec = random_vec/np.linalg.norm(random_vec)

        #   Take inner product of spins with random vector to compute parallel components
        parallel_component = np.sum(lattice*random_vec, axis=2)
        parallel_projection = parallel_component[:, :, None] * random_vec
        perpendicular_projection = lattice - parallel_projection

        #   Create lattice of + or - spins depending on paralell component
        spins = np.zeros(shape=[N, N]) 
        spins[parallel_component < 0] = -1 
        spins[parallel_component > 0] = 1

        #   Now treat the spin exactly like the Ising model
        links = find_links(N, spins, 1/T, parallel_component = parallel_component)
        cluster_labels, label_list, cluster_count = find_clusters(N, links)
        new_spins = assign_new_cluster_spins(N, cluster_labels, label_list)

        #   Flip parallel part of spins
        parallel_projection[spins != new_spins, :] = - parallel_projection[spins != new_spins, :]
        lattice = parallel_projection + perpendicular_projection

         # Animation. Doesn't work very well, you have to close the window at each iteration
        # Q2 = plt.quiver(lattice[:, :, 0], lattice[:, :, 1])
        # l, r, b, t = plt.axis()
        # dx, dy = r - l, t - b
        # plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])
        # plt.show()

        if (i+1) % n_idle == 0:
            print(np.mean(angles))
            helicity_modulus.append((T, compute_helicity_modulus(N, angles, T)))
            angles = np.zeros(shape = [n_idle, N, N])
            T = T + T_step

            print("T: " + str(T))

    return helicity_modulus

if __name__ == '__main__':
    #   Default simulation parameters
    N = 40
    T_init = 0.7
    T_end = 1.5
    T_step = 0.1
    n_idle = 100

    helicity_modulus = simulate(N, T_init, T_end, T_step, n_idle)

    plt.scatter(*zip(*helicity_modulus))
    plt.show()