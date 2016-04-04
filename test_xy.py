import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import math
import itertools
from numba import jit
import time

N=4

lattice = np.random.uniform(-1, 1, size = [N, N, 2])
lattice = lattice/np.linalg.norm(lattice, axis=2, keepdims=True)    
plt.figure()

Q = plt.quiver(lattice[:, :, 0], lattice[:, :, 1])
l, r, b, t = plt.axis()
dx, dy = r - l, t - b
plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])

random_vec = np.array([1,0])
parallel_component = np.sum(lattice*random_vec, axis=2)

parallel_projection = parallel_component[:, :, None] * random_vec
perpendicular_projection = lattice - parallel_projection

spins = np.zeros(shape=[N, N]) 
spins[parallel_component < 0] = -1 
spins[parallel_component > 0] = 1

new_spins = -spins

parallel_projection[spins != new_spins, :] = - parallel_projection[spins != new_spins, :]
new_lattice = parallel_projection + perpendicular_projection

Q2 = plt.quiver(new_lattice[:, :, 0], new_lattice[:, :, 1])
l, r, b, t = plt.axis()
dx, dy = r - l, t - b
plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])

print(lattice)
print(new_lattice)
plt.show()  