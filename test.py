from simulate_par import link, latti_upd
import numpy as np

a = np.matrix([[1,1,0],[1,0,1],[1,1,1]])
print(a)
links = link(3, a, 100)
a = latti_upd(3, a, links)
print(a)