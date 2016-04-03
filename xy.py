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

N=2

#   Create random spins and normalize
lattice = np.random.uniform(-1, 1, size = [N, N, 3])

lattice = np.array(
                    [
                        [[1,0,0], [5, 0, 0]],
                        [[-3, 4, 0], [0, 2, 0]]
                    ]
                    )
lattice = lattice/np.linalg.norm(lattice, axis=2, keepdims=True)

#   Create random vector and normalize
# random_vec = np.random.uniform(-1, 1, size = 3)
random_vec = np.array([2,0,0])
random_vec = random_vec/np.linalg.norm(random_vec)

#   Compute parallel and perpendicular components
parallel_component = np.sum(lattice*random_vec, axis=2)

spins = np.zeros(shape=[N, N]) 
spins[parallel_component < 0] = -1 
spins[parallel_component > 0] = 1

parallel_projection = parallel_component[:, :, None] * random_vec
perpendicular_projection = lattice - parallel_projection

find_links(N, spins, parallel_component, 0.5)