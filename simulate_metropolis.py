import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage import convolve, generate_binary_structure, iterate_structure
import math
from pprint import pprint

def sum_values_at_distance(a, i, j):
    sum_of_neighbours =  np.roll(np.roll(a, i, 0), j, 1) + np.roll(np.roll(a, i, 0), -j, 1) + np.roll(np.roll(a, -i, 0), j, 1) + np.roll(np.roll(a, -i, 0),-j, 1)
    return np.mean(a*sum_of_neighbours/4)

def compute_energy(lattice):
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    neighbour_sum = convolve(lattice, k, mode='wrap')
    energy = -0.5*(np.sum(neighbour_sum*lattice))

    return energy

def flipping_probabilities(N, lattice, mask, betaJ):
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    neighbour_sum = convolve(lattice, k, mode='wrap')  

    flipping_probability = np.zeros([N, N])
    flipping_probability[mask] = np.exp(-2*betaJ*lattice[mask]*neighbour_sum[mask]) - np.random.rand(N, N)[mask]

    return flipping_probability

def simulate(N, n_iter, betaJ, anim_params):
    lattice = np.ones([N, N])
    
    #   Checkerboard pattern to flip spins
    checkerboard = np.full((N, N), True, dtype=bool)
    checkerboard[1::2,::2] = False
    checkerboard[::2,1::2] = False
   
    #   Physical quantities to track
    susceptibility = {}
    binder_cumulant = {}
    magnetization = {}
    energy = {}
   
    if anim_params['animate']:
        plt.ion()
        fig = plt.figure()

    for i in range(n_iter):
        #   Compute probabilities and flip for one pattern of checkerboard
        flip_probs = flipping_probabilities(N, lattice, checkerboard, betaJ)
        lattice[flip_probs>0]=-lattice[flip_probs>0]

        #   Compute probabilities and flip for the other pattern of checkerboard
        flip_probs = flipping_probabilities(N, lattice, ~checkerboard, betaJ)
        lattice[flip_probs>0]=-lattice[flip_probs>0]

        #   Save physical quantities
        if betaJ in magnetization:
            magnetization[betaJ] = np.append(magnetization[betaJ], np.mean(lattice))
            energy[betaJ] = np.append(energy[betaJ], compute_energy(lattice))
        else:
            magnetization[betaJ] = np.array([np.mean(lattice)])
            energy[betaJ] = np.array([compute_energy(lattice)])

        if anim_params['animate'] and i%anim_params['freq'] == 0:
            fig.clf()
            ax = fig.add_subplot(111)
            ax.matshow(lattice)
            plt.draw()

        # if abs(betaJ*J-0.40)<0.005:
            # corr = np.array( [ [ sum_values_at_distance(lattice, i, j) for j in range(N)] for i in range(N)] )

        if i%100==0:           
            betaJ -= 0.01
            print("beta*J " + str(betaJ))
            
    return magnetization, energy

if __name__ == '__main__':    
    #   Default simulation parameters
    N = 128 
    n_iter = 10000
    betaJ = 1

    anim_params = {'animate': False, 'freq': 100}

    magnetization, energy = simulate(N, n_iter, betaJ, anim_params)

    cv = [(betaJ, ( betaJ**2*(np.var(energy[betaJ]) - np.std(energy[betaJ])) )/N**2) for betaJ in energy]
    plt.scatter(*zip(*cv))
    plt.show()