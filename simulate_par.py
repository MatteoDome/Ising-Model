import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.ndimage import convolve
import random

#   Simulation parameters
N = 15
n_iter = 10000
betaJ = 1

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

largest_label = 0;
label_hk = np.zeros([N, N])

def label(N, lattice_hk, label_hk):
    largest_label = 0
    label_hk = np.zeros([N, N])
    for i in range (0, N-1):
        for j in range (0, N-1):
            if lattice_hk[i,j] == lattice_hk[i, (j-1)%N]:
                label_hk[i,j] = label_hk[i, (j-1)%N]
            elif lattice_hk[i,j] == lattice_hk[(i-1)%N, j] and lattice_hk[(i-1)%N, j] != lattice_hk[i, (j-1)%N]:
                label_hk[i,j] = label_hk[(i-1)%N, j]
            else:
                largest_label = largest_label + 1
                label_hk[i,j] = largest_label

    for i in range (0, N-1):
        for j in range (0, N-1):
            if label_hk[i,j] != label_hk[i%N, (j+1)%N] and lattice_hk[i,j] == lattice_hk[i%N, (j+1)%N]:
                label_hk[i, j] = min(label_hk[i,j], label_hk[i%N, (j+1)%N])
            elif label_hk[i,j] != label_hk[(i+1)%N, j%N] and lattice_hk[i,j] == lattice_hk[(i+1)%N, j%N]:
                label_hk[i, j] = min(label_hk[i,j], label_hk[(i+1)%N, j%N])
    return label_hk, largest_label

def new_lattice(N, lattice_hk, label_hk, largest_label):
    new_lattice = np.zeros([N, N])
    new_spin = np.zeros([largest_label, 1])
    for i in range(0, largest_label):
        new_spin[i] = random.choice([-1, 1])
    for i in range (0, N):
        for j in range(0, N):
            new_lattice[i, j] = new_spin[label_hk[i,j]]
    return new_lattice

label_hk, largest_label = label(N, lattice_hk, label_hk)
new_lattice = new_lattice(N, lattice_hk, label_hk, largest_label)
print(new_lattice)
print(label_hk)



# for x in 0 to n_columns {
#  for y in 0 to n_rows {
#    if occupied[x,y] then
#      left = occupied[x-1,y];
#      above = occupied[x,y-1];
#      if (left == 0) and (above == 0) then        /* Neither a label above nor to the left. */
#        largest_label = largest_label + 1;        /* Make a new, as-yet-unused cluster label. */
#        label[x,y] = largest_label;

#      else if (left != 0) and (above == 0) then   /* One neighbor, to the left. */
#        label[x,y] = find(left); 

#      else if (left == 0) and (above != 0) then   /* One neighbor, above. */
#        label[x,y] = find(above);

#      else                                        /* Neighbors BOTH to the left and above. */
#        union(left,above);                        /* Link the left and above clusters. */
#        label[x,y] = find(left);
#      }
#    }
#  }
