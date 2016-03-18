import numpy as np
from scipy.ndimage import convolve, generate_binary_structure, iterate_structure

def correlation(lattice, i, j):
    """
    MAKE THIS CLEARER
    """

    sum_of_neighbours = np.roll(np.roll(lattice, i, 0), j, 1) + np.roll(np.roll(lattice, i, 0), -j, 1) + np.roll(np.roll(lttice, -i, 0), j, 1) + np.roll(np.roll(lattice, -i, 0),-j, 1)
    
    return np.mean(a*sum_of_neighbours/4)

def compute_energy(lattice):
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    neighbour_sum = convolve(lattice, k, mode='wrap')
    energy = -0.5*(np.sum(neighbour_sum*lattice))

    return energy