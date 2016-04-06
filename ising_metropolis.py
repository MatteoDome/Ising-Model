import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure, iterate_structure

def flipping_probabilities(N, lattice, neighbour_list, spins_to_flip, betaJ):
    neighbour_sum = convolve(lattice, neighbour_list, mode='wrap')

    flipping_probability = np.zeros([N, N])
    flipping_probability[spins_to_flip] = np.exp(-2 * betaJ * lattice[spins_to_flip] * neighbour_sum[
                                        spins_to_flip]) - np.random.rand(N, N)[spins_to_flip]

    return flipping_probability

def simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle, second_neighbours):
    #   Simulation variables
    lattice = np.random.choice([1, -1], size=[N, N])
    betaJ = betaJ_init
    n_iter = int((betaJ_end - betaJ_init) / betaJ_step * n_idle)
    
    #   Physical quantities to track
    magnetization = { betaJ_init : np.array([])}
    energy = { betaJ_init : np.array([]) }
    lat_sum = { betaJ_init : np.array([]) }

    #   Patterns we use to flip spins
    if second_neighbours:
        neighbour_list = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    
        patterns = [ np.full((N, N), False, dtype=bool), 
                     np.full((N, N), False, dtype=bool), 
                     np.full((N, N), False, dtype=bool), 
                     np.full((N, N), False, dtype=bool)]

        patterns[0][::2, ::2] = True
        patterns[1][::2, 1::2] = True
        patterns[2][1::2, ::2] = True
        patterns[3][1::2, 1::2] = True

    else:
        neighbour_list = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
        patterns = [ np.full((N, N), False, dtype=bool), []]

        patterns[0][1::2, ::2] = True
        patterns[0][::2, 1::2] = True
        patterns[1] = ~patterns[0]


    #   Main cycle
    for i in range(n_iter):
        #   Compute probabilities for different patterns of the lattice
        for pattern in patterns:
            flip_probs = flipping_probabilities(N, lattice, neighbour_list, pattern, betaJ)
            lattice[flip_probs > 0] = -lattice[flip_probs > 0]

        #   Save physical quantities
        neighbour_sum = convolve(lattice, neighbour_list, mode='wrap')
        energy[betaJ] = np.append(energy[betaJ], -0.5 * (np.sum(neighbour_sum * lattice)))
        magnetization[betaJ] = np.append(magnetization[betaJ], np.mean(lattice))
        lat_sum[betaJ] = np.append(lat_sum[betaJ], np.sum(lattice))

        if i % n_idle == 0:
            betaJ = betaJ + betaJ_step
            
            magnetization[betaJ] = np.array([])
            energy[betaJ] = np.array([]) 
            lat_sum[betaJ] =  np.array([])
            
            print("beta*J " + str(betaJ))

    #   Process data
    magnetization = [(betaJ, np.mean(magnetization[betaJ])) for betaJ in magnetization]
    susceptibility = [(betaJ, (np.mean(lat_sum[betaJ]**2)-(np.mean(abs(lat_sum[betaJ]))**2))/N**2) for betaJ in lat_sum]
    binder_cumulant = [(betaJ, 1 - np.mean(lat_sum[betaJ]**4) /
                        (3 * np.mean(lat_sum[betaJ]**2)**2)) for betaJ in lat_sum]
    cv = [(betaJ, (betaJ**2 * (np.var(energy[betaJ]))) / N**2) for betaJ in energy]

    return magnetization, susceptibility, binder_cumulant, cv

if __name__ == '__main__':
    #   Default simulation parameters
    N = 4
    betaJ_init = 0.01
    betaJ_end = 1
    betaJ_step = 0.01
    n_idle = 10000
    second_neighbours = False

    magnetization, susceptibility, binder_cumulant, cv = simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle, second_neighbours)

    plt.scatter(*zip(*magnetization))
    plt.show()