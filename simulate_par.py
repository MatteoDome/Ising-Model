import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage import convolve
import random
import math
from numba import jit

#   Simulation parameters
N = 50
T = 1
n_iter = 10000
n_iter_init = 1000
betaJ = 0.01
E= np.zeros([n_iter - n_iter_init])
#   System variables
#lattice = np.ones([N, N])


# #   Checkerboard pattern to flip spins
# indices = np.full((N, N), True, dtype=bool)
# indices[1::2,::2] = False
# indices[::2,1::2] = False

# #   This is used for the spin sum
# k = np.array([[0,1,0],[1,0,1],[0,1,0]])

# plt.ion()
# fig = plt.figure()

# magnetization = np.zeros(1000)
# x = np.zeros(1000)

# for i in range(n_iter):
#     delta_energy = 2*betaJ*lattice*convolve(lattice, k, mode='wrap')  
#     flipping_probability = np.exp(-delta_energy)-np.random.rand(N, N)

#     #   Clean
#     if i%2==0:
#         flipping_probability[indices] = 0
#     else:
#         flipping_probability[~indices] = 0

#     lattice[flipping_probability>0]=-lattice[flipping_probability>0]
#     # if(betaJ<0.5):
#     #     fig.clf()
#     #     ax = fig.add_subplot(111)
#     #     ax.matshow(lattice)
#     #     plt.draw()

#     if i%100==0:
#         x[int(i/100)] = betaJ
#         magnetization[int(i/100)] = np.mean(lattice)
#         betaJ-=0.01
#         print("BETAJ " + str(betaJ))

#     # print(i)

# plt.plot(x,magnetization)
# plt.show()


##Hoshen Kopelman


lattice_hk = np.zeros([N,N])
for i in range (0,N):
    for j in range (0, N):
        lattice_hk[i,j] = random.choice([-1, 1])
largest_label = 0
label_hk = np.zeros([N, N])
links_hk= np.zeros([N, N, 2])

@jit
def link(N, lattice_hk, links_hk, betaJ):
    prob = np.exp(-2*betaJ)
    for i in range (0, N):
        for j in range (0, N):
            if lattice_hk[i,j] == lattice_hk[(i-1)%N, j]:
                if np.random.uniform(0,1) < prob:
                    links_hk[i,j,0] = 0
                else:
                    links_hk[i,j,0] = 1

            if lattice_hk[i,j] == lattice_hk[i, (j-1)%N]:
                if np.random.uniform(0,1) < prob:
                    links_hk[i,j,1] = 0
                else:
                    links_hk[i,j,1] = 1
    return links_hk
# @jit
# def label(N, lattice_hk, links_hk, betaJ):
#     largest_label = 0 
#     label_hk = np.zeros([N, N])
#     for i in range (0, N):
#         for j in range (0, N):         
#             if links_hk[i,j,0] == 1:                        #can be done in one if
#                 label_hk[i,j] = label_hk[(i-1)%N, j]
#             if links_hk[i,j,1] == 1:
#                 label_hk[i,j] = label_hk[i, (j-1)%N]
#             elif all(links_hk[i,j,:]) != 1:
#                 largest_label = largest_label + 1
#                 label_hk[i,j] = largest_label

#     for i in range (0, N):
#         for j in range (0, N):
#             if links_hk[(i+1)%N,j,0] ==1 and label_hk[i,j] != label_hk[(i+1)%N, j]:
#                 label_hk[i, j] = min(label_hk[i,j], label_hk[(i+1)%N, j])
#             if links_hk[i,(j+1)%N,1] ==1 and label_hk[i,j] != label_hk[i, (j+1)%N]:
#                 label_hk[i, j] = min(label_hk[i,j], label_hk[i, (j+1)%N])

#     return label_hk, largest_label

# @jit
# def new_lattice(N, lattice_hk, label_hk, largest_label):

#     up_lattice = np.zeros([N, N])
#     new_spin = np.zeros([largest_label])
#     for i in range(0, largest_label):
#         new_spin[i] = random.choice([-1, 1])
#     for i in range (0, N):
#         for j in range(0, N):
#             up_lattice[i, j] = new_spin[(label_hk[i,j])]
#     return up_lattice


def latti_upd(N, lattice_hk, links_hk, betaJ):
    largest_label = 0 
    label_hk = np.zeros([N, N])
    for i in range (0, N):
        for j in range (0, N):         
            if links_hk[i,j,0] == 1 and links_hk[i,j,1] != 1:                        #can be done in one if
                label_hk[i,j] = label_hk[(i-1)%N, j]
            if links_hk[i,j,1] == 1 and links_hk[i,j,0] != 1:
                label_hk[i,j] = label_hk[i, (j-1)%N]
            elif links_hk[i,j,1] == 1 and links_hk[i,j,0] == 1:
                label_hk[i,j] = label_hk[(i-1)%N, j]
            elif all(links_hk[i,j,:]) != 1:
                largest_label += 1
                label_hk[i,j] = largest_label
    print("----")
    print(largest_label)
    for i in range (0, N):
        for j in range (0, N):
            if links_hk[(i+1)%N,j,0] ==1 and label_hk[i,j] != label_hk[(i+1)%N, j]:
                label_hk[i, j] = min(label_hk[i,j], label_hk[(i+1)%N, j])
            if links_hk[i,(j+1)%N,1] ==1 and label_hk[i,j] != label_hk[i, (j+1)%N]:
                label_hk[i, j] = min(label_hk[i,j], label_hk[i, (j+1)%N])

    largest_label = largest_label +1
    up_lattice = np.zeros([N, N])
    new_spin = np.zeros([largest_label])
    for i in range (0, largest_label):
            new_spin[i] = random.choice([-1, 1])
    for i in range (0, N):
        for j in range(0, N):
            up_lattice[i, j] = new_spin[(label_hk[i,j])]
    return up_lattice

@jit
def energy_cal(lattice_hk, betaJ):
    E = 0
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    for i in range (0,N):
        for j in range (0,N):
            E = E-0.5*betaJ*np.sum(lattice*convolve(lattice_hk, k, mode='wrap'))
    return E

lattice_sum = np.zeros(1000)
x = np.zeros(1000)
magnetization = np.zeros(1000)
for i in range (0, n_iter):
    link(N, lattice_hk, links_hk, betaJ)
    lattice_hk = latti_upd(N, lattice_hk, links_hk, betaJ)
    lattice_sum[i%100] = lattice_hk.sum()
    if i%100==0:
        x[int(i/100)] = betaJ
        magnetization[int(i/100)] = abs(np.mean(lattice_sum))
        betaJ+=0.01
        lattice_sum = np.zeros(100)


    # print(i)

plt.plot(x,magnetization)
plt.show()
    # # if i > n_iter_init:
    # #     E[i-n_iter_init] = energy_cal(lattice_hk, betaJ)/N
    # # print("-------")
    # print(i)

# Cv = np.var(E)*betaJ/(N*N)
# print(Cv)