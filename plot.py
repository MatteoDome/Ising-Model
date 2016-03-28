from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt
import numpy as np

betaJ_values = np.arange(0.1, 1, 0.1)
N_values = [4, 8, 16, 32, 64, 128, 256]

with PdfPages('batch_results/magnetization.pdf') as pdf:
    plt.figure()
    for N in N_values:
    	results = open("batch_results/"+str(N), 'rb')
    	object_results = pickle.load(results)
    	for betaJ in betaJ_values:
		    plt.plot(betaJ, object_results.magnetization[0], linewidth=2.0)

		    plt.xlabel('$betaJ$',fontsize=15)
		    plt.xlim([0, 1])
		    plt.ylabel('$magnetization$', fontsize=15)
		    plt.legend(loc=4)

		    pdf.savefig() 
plt.close()

file.close()