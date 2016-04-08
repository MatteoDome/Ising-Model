from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

lattice_sizes, suscept_pos = zip(*[(4., 0.17853), (8., 0.3156), (16., 0.377971), (32., 0.41023), (64., 
  0.425546)])
suscept_pos_error = [0.0006346, 0.0003554, 0.000862583, 0.00025, 0.00021]

_, suscept_pos_sec = zip(*[(4., 0.140144), (8., 0.166519), (16., 0.178613), (32., 
  0.18383), (64., 0.1867)])
suscept_pos_sec_error = [0.00029, 0.000259, 0.00017583, 0.0003149, 0.000432]

_, suscept_pos_third = zip(*[[4., 0.52851], [8., 0.642346], [16., 0.69354], [32., 0.71767], [64., 
  0.72953]])
suscept_pos_third_error = [0.00043, 0.000488, 0.0005567, 0.000657, 0.000701]


x = np.arange(0.1, 80, 0.01);
suscept_pos_expr = -1.14719*(x)**(-1.07181) + 0.438574 
suscept_pos_sec_expr = -0.234068*(x)**(-1.13605) + 0.188599
suscept_pos_third_expr = -0.994618*(x)**(-1.12291) + 0.738271

with PdfPages('plot_pos.pdf') as pdf:

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$L$', fontsize=20)
    ax.set_ylabel('$\\chi_{position}$', fontsize=20)

    axes = plt.gca()

    axes.set_color_cycle(['red', 'blue', 'green'])
    axes.set_xlim([0.5, 80])
    axes.set_ylim([0, 1.1])

    plt.errorbar(lattice_sizes, suscept_pos, yerr=suscept_pos_error, fmt='o', label='$1^{st}$ neighbours, $\\beta J_c = 0.4385(1)$, $\\nu = 1.07(1)$')
    plt.errorbar(lattice_sizes, suscept_pos_sec, yerr=suscept_pos_sec_error, fmt='v', label='$2^{nd}$ neighbours (ferromagnetic), $\\beta J_c = 0.1885(2)$, $\\nu = 1.13(1)$')
    plt.errorbar(lattice_sizes, suscept_pos_third, yerr=suscept_pos_third_error, fmt='v', label='$2^{nd}$ neighbours (antiferromagnetic), $\\beta J_c = 0.7382(8)$, $\\nu = 1.12(1)$')
    plt.legend(numpoints=1, fontsize = 'medium')
    plt.plot(x, suscept_pos_expr)
    plt.plot(x, suscept_pos_sec_expr)
    plt.plot(x, suscept_pos_third_expr)


    pdf.savefig()
    plt.close()
