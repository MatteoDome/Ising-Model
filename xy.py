import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import math
import itertools
from numba import jit

N=2

#   Create random spins and normalize
lattice = np.random.uniform(-1, 1, size = [N, N, 3])
lattice /= np.linalg.norm(lattice, axis=2, keepdims=True)

#   Create random vector and normalize
random_vec = np.random.uniform(-1, 1, size = 3)
random_vec /= np.linalg.norm(random_vec)

#   Compute parallel and perpendicular components
dot_product = np.sum(lattice*random_vec, axis=2)
parallel_projection = dot_product[:, :, None] * random_vec
perpendicular_projection = lattice - parallel_projection

