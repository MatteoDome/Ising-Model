from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt
import numpy as np


lattice_sizes, magnetization = zip(*[(8, 0.90404375), (16, 0.841697265625), (32, 0.774832519531), (64, 
  0.711084960937)])

x = np.arange(0, 4.5, 0.01);
magnetization_expr = 0.144043 - 0.115853*x

with PdfPages('plot_magnetization.pdf') as pdf:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$N$', fontsize=20)
    ax.set_ylabel('$m(\\beta J_c$)', fontsize=20)

    axes = plt.gca()
    axes.set_xlim([0.5, 4.5])

    plt.scatter(np.log(lattice_sizes), np.log(magnetization), label='Nearest neighbours, $\\beta = 0.115(3)$')
    plt.plot(x, magnetization_expr)

    plt.legend(loc='upper right')
    pdf.savefig()
