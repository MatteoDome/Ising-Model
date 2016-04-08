import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from numba import jit
from cluster_finding import canonical_label, link_labels, find_clusters, assign_new_cluster_spins, find_links

def plot_lattice(N, lattice):
    plot_latt = np.zeros(shape=[N, N , 2])
    plot_latt[:, :, 0] = np.cos(lattice[:, :])
    plot_latt[:, :, 1] = np.sin(lattice[:, :])
    Q2 = plt.quiver(plot_latt[:, :, 0], plot_latt[:, :, 1])
    l, r, b, t = plt.axis()
    dx, dy = r - l, t - b
    plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])
    plt.show()

def compute_helicity_modulus(N, angles, T):
    a = np.cos(angles - np.roll(angles, 1, axis = 2)) + \
        np.cos(angles - np.roll(angles, 1, axis = 1)) 

    a = np.sum(a, axis = (1, 2))

    b = np.sin(angles - np.roll(angles, 1, axis=2))
    b = np.sum(b, axis = (1, 2))
    
    c = np.sin(angles - np.roll(angles, 1, axis=1))
    c = np.sum(c, axis = (1, 2))
    helicity = np.abs(1/(2*N**2)*(np.mean(a) - np.mean(b**2)/T - np.mean(c**2)/T))

    return helicity

def simulate(N, T_init, T_end, T_step, n_idle):
    lattice = np.zeros(shape = [N, N]) 
    T = T_init
    n_iter = int((T_end - T_init) / T_step * n_idle)
    neighbour_list = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    #   Physical quantities to track
    helicity_modulus = []
    angles = np.zeros(shape = [n_idle, N, N])
    
    plot_lattice(N, lattice)

    #   Main cycle
    for i in range(n_iter):
        #   Store angles of the spins to later calculate helicity modulus
        angles[int(i%n_idle), :, :] = lattice
        random_angle = np.random.uniform(0, 2*math.pi)
        parallel_component = np.cos(lattice - random_angle)
        
        #   Create lattice of + or - spins depending on paralell component
        spins = np.zeros(shape=[N, N]) 
        spins[parallel_component < 0] = -1 
        spins[parallel_component > 0] = 1

        #   Now treat the spin exactly like the Ising model
        links = find_links(N, spins, 1/T, parallel_component = parallel_component)
        cluster_labels, label_list, cluster_count = find_clusters(N, links)
        new_spins = assign_new_cluster_spins(N, cluster_labels, label_list)

        #   Flip parallel part of spins
        #print(random_angle/math.pi)
        #print(spins)
        lattice[spins != new_spins] = (lattice+2*(random_angle-lattice))[spins != new_spins]

        if (i+1) % n_idle == 0:
            helicity_modulus.append((T, compute_helicity_modulus(N, angles, T)))
            angles = np.zeros(shape = [n_idle, N, N])
            T = T + T_step

            print("T: " + str(T))

        #print(new_spins)
        plot_lattice(N, lattice)

    return helicity_modulus

if __name__ == '__main__':
    #   Default simulation parameters
    N = 20
    T_init = 0.1
    T_end = 3.2
    T_step = 0.1
    n_idle = 1000

    helicity_modulus = simulate(N, T_init, T_end, T_step, n_idle)

    plt.scatter(*zip(*helicity_modulus))
    plt.show()