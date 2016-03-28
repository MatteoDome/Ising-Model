from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt
import numpy as np

betaJ_values = np.arange(0.1, 1, 0.01)
N_values = [4, 8, 16, 32, 64, 128, 256]

with PdfPages('batch_results/magnetization.pdf') as pdf:
    plt.figure()
    for N in N_values:
    	results = open("batch_results/"+str(N), 'rb')
    	object_results = pickle.load(results)
    	magnetization = np.asarray(object_results[0])
    	for betaJ in betaJ_values:
    		betaJ_ind = int(round(betaJ*100 - 10))
    		magn_pl = magnetization[betaJ_ind,1]
    		print("*******")
    		print("betaJ = " + str(betaJ))
    		print("magnetization = " +str(magn_pl))
    		print("N = "+str(N))
    		plt.plot(betaJ, magn_pl, linewidth=2.0)
    	results.close()
	plt.xlabel('$betaJ$',fontsize=15)
	plt.xlim([0, 1])
	plt.ylabel('$magnetization$', fontsize=15)
	plt.legend(loc=4)
	plt.ylim([0,1])
	pdf.savefig()
    	

plt.close()