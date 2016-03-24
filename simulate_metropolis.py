import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure, iterate_structure

def compute_energy(lattice, neighbour_list):
    """

    Compute the energy of the lattice. neighbour_list is a matrix indicating which
    neighbour spins should be considered when computing the energy (1 contributes, 0
    does not).
   
    Args
        neighbour_list: 3x3 matrix indicating which neighbour interactions to consider
    """

    #   Wrap mode used for periodic boundary conditions
    neighbour_sum = convolve(lattice, neighbour_list, mode='wrap')
    energy = -0.5 * (np.sum(neighbour_sum * lattice))

    return energy

def flipping_probabilities(N, lattice, neighbour_list, mask, betaJ):
    neighbour_sum = convolve(lattice, neighbour_list, mode='wrap')

    flipping_probability = np.zeros([N, N])
    flipping_probability[mask] = np.exp(-2 * betaJ * lattice[mask] * neighbour_sum[
                                        mask]) - np.random.rand(N, N)[mask]

    return flipping_probability


def simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle):
    #   Simulation variables
    lattice = np.random.choice([1, -1], size=[N, N])
    betaJ = betaJ_init
    n_iter = int((betaJ_end - betaJ_init) / betaJ_step * n_idle)
    neighbour_list = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
    #   betaJ values that will be sweeped
    betaJ_values = [round(betaJ_init + i * betaJ_step, 2)
              for i in range(int((betaJ_end - betaJ_init) / betaJ_step) + 1)]

    #   Physical quantities to track
    magnetization = { betaJ : np.array([]) for betaJ in betaJ_values}
    energy = { betaJ : np.array([]) for betaJ in betaJ_values}
    lat_sum = { betaJ : np.array([]) for betaJ in betaJ_values}

    #   Checkerboard pattern to flip spins
    checkerboard = np.full((N, N), True, dtype=bool)
    checkerboard[1::2, ::2] = False
    checkerboard[::2, 1::2] = False

    #   Main cycle
    for i in range(n_iter):
        #   Compute probabilities and flip for one pattern of checkerboard
        flip_probs = flipping_probabilities(N, lattice, neighbour_list, checkerboard, betaJ)
        lattice[flip_probs > 0] = -lattice[flip_probs > 0]

        # Compute probabilities and flip for the other pattern of checkerboard
        flip_probs = flipping_probabilities(N, lattice, neighbour_list, ~checkerboard, betaJ)
        lattice[flip_probs > 0] = -lattice[flip_probs > 0]

        #   Save physical quantities
        magnetization[betaJ] = np.append(magnetization[betaJ], np.mean(lattice))
        energy[betaJ] = np.append(energy[betaJ], compute_energy(lattice, neighbour_list))
        lat_sum[betaJ] = np.append(lat_sum[betaJ], np.sum(lattice))

        if i % n_idle == 0:
            betaJ = round(betaJ + betaJ_step, 2)
            print("beta*J " + str(betaJ))

    #   Process data
    magnetization = [(betaJ, np.mean(magnetization[betaJ])) for betaJ in magnetization]
    susceptibility = [(betaJ, (np.mean(lat_sum[betaJ]**2)-(np.mean(abs(lat_sum[betaJ]))**2))/N**2) for betaJ in lat_sum]
    binder_cumulant = [(betaJ, 1 - np.mean(lat_sum[betaJ]**4) /
                        (3 * np.mean(lat_sum[betaJ]**2)**2)) for betaJ in lat_sum]
    cv = [(betaJ, (betaJ**2 * (np.var(energy[betaJ]))) / N**2) for betaJ in energy]

    return magnetization, susceptibility, binder_cumulant, cv

if __name__ == '__main__':
    #   Simulation parameters
    N = 100
    betaJ_init = 0.01
    betaJ_end = 1
    betaJ_step = 0.01
    n_idle = 200

    magnetization, susceptibility, binder_cumulant, cv = simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle)

    plt.scatter(*zip(*magnetization))
    # plt.scatter(*zip(*cv))
    plt.show()
