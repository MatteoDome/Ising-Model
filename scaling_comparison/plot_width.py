from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt
import numpy as np


lattice_sizes = np.log([4, 8, 16, 32, 64])

_, suscept_width = zip(*[(1.38629, -2.08265), (2.07944, -2.67233), (2.77259, -3.27995), \
(3.46574, -3.93529), (4.15888, -4.61026)])
suscept_width_error = [0.00638844, 0.00908367, 0.0125105, 0.0230348, 0.0362842]

_, suscept_width_sec = zip(*[(1.38629, -2.91692), (2.07944, -3.63746), (2.77259, -4.37724), \
(3.46574, -5.01932), (4.15888, -5.66418)])
suscept_width_sec_error = [0.0090573, 0.0159581, 0.0221338, 0.10985, 0.156286]

_, suscept_width_third = zip(*[[1.38629, -1.79854], [2.07944, -2.41531], [2.77259, -2.97436], \
[3.46574, -3.58679], [4.15888, -4.2454]])
suscept_width_third_error = [0.00694696, 0.0111932, 0.0172279, 0.0348539, 0.0605722]


x = np.arange(0, 4.5, 0.01);
suscept_width_expr = -0.882289*x - 0.852213
suscept_width_sec_expr = -1.04379*x -1.47014
suscept_width_third_expr = -0.604359 - 0.863273*x


with PdfPages('plot_width.pdf') as pdf:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\\log(L)$', fontsize=20)
    ax.set_ylabel('$\\log(\\chi_{width})$', fontsize=20)

    axes = plt.gca()

    axes.set_color_cycle(['red', 'blue', 'green'])
    # axes.set_xlim([0.5, 4.5])
    # axes.set_ylim([-1, 5.5])

    plt.errorbar(lattice_sizes, suscept_width, yerr=suscept_width_error, fmt='o', label='$1^{st}$ neighbours, $\\nu = 1.07(1)$')
    plt.errorbar(lattice_sizes, suscept_width_sec, yerr=suscept_width_sec_error, fmt='v', label='$2^{nd}$ neighbours (ferromagnetic), $\\nu = 1.13(1)$')
    plt.errorbar(lattice_sizes, suscept_width_third, yerr=suscept_width_third_error, fmt='v', label='$2^{nd}$ neighbours (antiferromagnetic), $\\nu = 1.15(1)$')
    plt.plot(x, suscept_width_expr)
    plt.plot(x, suscept_width_sec_expr)
    plt.plot(x, suscept_width_third_expr)

    plt.legend(loc='lower left', numpoints=1, fontsize = 'medium')
    pdf.savefig()
