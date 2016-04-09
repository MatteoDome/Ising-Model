from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

T_values = np.arange(0.7, 1.5, 0.1)
N_values = [8, 12, 20, 30, 40, 100]
colors = ['blue', 'green', 'red', 'cyan', 'purple', 'gold']
fmt = ['D', '+', 's', 'x', '^', '*']
yticks = np.arange(0, 1, 0.01)
with PdfPages('batch_xy/helicity_modulus.pdf') as pdf:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for index, N in enumerate(N_values):
        results = open("batch_xy/"+str(N), 'rb')
        helicity_modulus = pickle.load(results)
        # plt.scatter(*zip(*helicity_modulus), label='N = ' + str(N), color = colors[index], marker=fmt[index])
        plt.plot(*zip(*helicity_modulus), label='L = ' + str(N), marker=fmt[index])
        results.close()
        plt.xlabel('$ k_B T/ J$',fontsize=15)
    
    ax.set_yticks(np.arange(0, 1, 0.1))      
    ax.set_yticks(yticks, minor=True)                                                                                            
    x = np.arange(0.6, 1.45, 0.1);
    line = 2*x/math.pi
    plt.plot(x, line)

    plt.xlim([0.6,1.55])
    plt.ylabel("$\\Gamma / J$", fontsize = 15)
    plt.legend()
    pdf.savefig()
    plt.show()

   
plt.close()
