import simulate_hk

N=4
betaJ_init = 0.1
betaJ_end = 0.6
betaJ_step = 0.01
n_idle = 10000

magnetization, susceptibility, binder_cumulant, cv = simulate_hk.simulate(N, betaJ_init, betaJ_end, betaJ_step, n_idle)

pickle.dump(results, open('batch_results/' + str(N),'wb'))

magnetization[betaJ]['T'].append(T)
magnetization[betaJ]['y'].append(results['magnetization']['value'])

cv[betaJ]['T'].append(T)
cv[betaJ]['y'].append(results['cv']['value'])

susceptibility[betaJ]['T'].append(T)
susceptibility[betaJ]['y'].append(results['susceptibility']['value'])


binder_cumulant[betaJ]['T'].append(T)
binder_cumulant[betaJ]['y'].append(results['binder_cumulant']['value'])