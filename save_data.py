import simulate_hk
import pickle

N=256
betaJ_init = 0.1
betaJ_end = 1
betaJ_step = 0.01
n_idle = 1000

magnetization, susceptibility, binder_cumulant, cv = simulate_hk.simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle)

my_list = [magnetization, susceptibility, binder_cumulant, cv]

pickle.dump(my_list, open('batch_results/' + str(N),'wb'))