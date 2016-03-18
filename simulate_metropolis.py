import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage import convolve, generate_binary_structure, iterate_structure
import physical_quantities as pq

def flipping_probabilities(N, lattice, mask, betaJ):
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    neighbour_sum = convolve(lattice, k, mode='wrap')  

    flipping_probability = np.zeros([N, N])
    flipping_probability[mask] = np.exp(-2*betaJ*lattice[mask]*neighbour_sum[mask]) - np.random.rand(N, N)[mask]

    return flipping_probability

def simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle, anim_params):
    #   Compute number of iterations
    n_iter = int((betaJ_end-betaJ_init)/betaJ_step*n_idle)

    #   Value os betaJ that are going to be tracked
    values = [round(betaJ_init + i*betaJ_step, 2) for i in range(int((betaJ_end-betaJ_init)/betaJ_step)+1)]
    
    #   Physical quantities to track
    magnetization = dict((betaJ, np.array([])) for betaJ in values)
    energy = dict((betaJ, np.array([])) for betaJ in values)
    l_sum = dict((betaJ, np.array([])) for betaJ in values)
    susceptibility = dict((betaJ, np.array([])) for betaJ in values)
    cv = dict((betaJ, np.array([])) for betaJ in values)

    #   Main lattice matrix and betaJ
    lattice = np.random.choice([1, -1], size = [N, N])
    betaJ = betaJ_init
    
    #   Checkerboard pattern to flip spins
    checkerboard = np.full((N, N), True, dtype=bool)
    checkerboard[1::2,::2] = False
    checkerboard[::2,1::2] = False
      
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
        l_sum[betaJ] = np.append(l_sum[betaJ], np.sum(lattice))
        energy[betaJ] = np.append(energy[betaJ], pq.compute_energy(lattice))
        susceptibility[betaJ] = np.append(susceptibility[betaJ], np.mean(lattice)**2)

        if anim_params['animate'] and i%anim_params['freq'] == 0:
            fig.clf()
            ax = fig.add_subplot(111)
            ax.matshow(lattice)
            plt.draw()

        # if abs(betaJ*J-0.40)<0.005:
            # corr = np.array( [ [pq.correlation(lattice, i, j) for j in range(N)] for i in range(N)] )

        if i%n_idle == 0:           
            betaJ = round(betaJ + betaJ_step, 2)
            print("beta*J " + str(betaJ))
            
    return energy, l_sum

if __name__ == '__main__':    
    #   Simulation parameters
    N = 32
    betaJ_init = 0.01
    betaJ_end = 1
    betaJ_step = 0.01
    n_idle = 100

    anim_params = {'animate': False, 'freq': 100}

    energy, l_sum = simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle, anim_params)

    cv = [(betaJ, ( betaJ**2*(np.var(energy[betaJ]) - np.std(energy[betaJ])) )/N**2) for betaJ in energy]
    binder_cumulant = [( betaJ, 1 - np.mean(l_sum[betaJ]**4)/( 3*np.mean(l_sum[betaJ]**2)**2 ) ) for betaJ in l_sum]
    magnetization = [( betaJ, np.mean(l_sum[betaJ])/N**2) for betaJ in l_sum]
    
    plt.scatter(*zip(*magnetization))
    #plt.scatter(*zip(*cv))
    plt.show()