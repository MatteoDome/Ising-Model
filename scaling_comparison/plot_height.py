from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt
import numpy as np


lattice_sizes = np.log([4, 8, 16, 32, 64])

_, suscept_height = zip(*[(1.38629, 0.360285), (2.07944, 1.44348), (2.77259, 
  2.58064), (3.46574, 3.75143), (4.15888, 4.95536)])
suscept_height_error = [0.00349701, 0.00360367, 0.00768614, 0.0090597, 0.0165292]

_, suscept_height_sec = zip(*[(1.38629, 0.270498), (2.07944, 1.38697), (2.77259, 2.56133), (3.46574,
   3.72735), (4.15888, 4.93169)])
suscept_height_sec_error = [0.0039676, 0.00726361, 0.00815255, 0.0294378, 0.104947]

x = np.arange(0, 4.5, 0.01);
suscept_height_sec_expr = 1.64854*x -2.01864
suscept_height_expr = 1.6267*x -1.91352

with PdfPages('plot_height.pdf') as pdf:

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$N$', fontsize=20)
    ax.set_ylabel('$\\chi_{height}$', fontsize=20)

    axes = plt.gca()

    axes.set_color_cycle(['red', 'blue'])
    axes.set_xlim([0.5, 4.5])
    axes.set_ylim([-1, 5.5])

    plt.errorbar(lattice_sizes, suscept_height, yerr=suscept_height_error, fmt='o', label='Nearest neighbour, $\\frac{\\gamma}{\\nu} = 1.62(2)$')
    plt.errorbar(lattice_sizes, suscept_height_sec, yerr=suscept_height_sec_error, fmt='v', label='Second nearest neighbours, $\\frac{\\gamma}{\\nu} = 1.64(1)$')
    plt.plot(x, suscept_height_expr)
    plt.plot(x, suscept_height_sec_expr)

    plt.legend(loc='upper left')

    pdf.savefig()
