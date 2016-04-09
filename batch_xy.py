from xy_hk import simulate
import pickle

#   Default simulation parameters
N = 100
T_init = 0.7
T_end = 1.6
T_step = 0.1
n_idle = 1000

helicity_modulus = simulate(N, T_init, T_end, T_step, n_idle)
pickle.dump(helicity_modulus, open('batch_xy/' + str(N),'wb'))
