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

_, suscept_height_third = zip(*[[1.38629, 0.471165], [2.07944, 1.63847], [2.77259, 
  2.84089], [3.46574, 4.03582], [4.15888, 5.22237]])
suscept_height_third_error = [0.00143583, 0.00343871, 0.00807892, 0.0181502, 0.0407977]

x = np.arange(0, 4.5, 0.01);
suscept_height_third_expr = -1.88601 + 1.69962*x
suscept_height_sec_expr = 1.64854*x -2.01864
suscept_height_expr = 1.6267*x -1.91352

with PdfPages('plot_height.pdf') as pdf:

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\\log(L)$', fontsize=20)
    ax.set_ylabel('$\\log(\\chi_[height])$', fontsize=20)

    axes = plt.gca()

    axes.set_color_cycle(['red', 'blue', 'green'])
    axes.set_xlim([0.5, 4.5])
    axes.set_ylim([-2, 5.5])

    plt.errorbar(lattice_sizes, suscept_height, yerr=suscept_height_error, fmt='o', label='$1^{st}$ neighbours, $\\gamma/\\nu = 1.62(2)$')
    plt.errorbar(lattice_sizes, suscept_height_sec, yerr=suscept_height_sec_error, fmt='v', label='$2^{nd}$ neighbours (ferromagnetic), $\\gamma/\\nu  = 1.64(1)$')
    plt.errorbar(lattice_sizes, suscept_height_third, yerr=suscept_height_third_error, fmt='v', label='$2^{nd}$ neighbours (antiferromagnetic), $\\gamma/\\nu  = 1.699(8)$')
    plt.plot(x, suscept_height_expr)
    plt.plot(x, suscept_height_sec_expr)
    plt.plot(x, suscept_height_third_expr)

    plt.legend(loc='lower right', numpoints=1, fontsize = 'medium')

    pdf.savefig()
