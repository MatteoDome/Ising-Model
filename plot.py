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
    	plt.scatter(magnetization[:,0], magnetization[:,1], c = np.random.rand(3,1))
    	results.close()
    	
    	plt.hold(True)
    plt.xlabel('$betaJ$',fontsize=15)
    plt.xlim([0,1])
    plt.ylabel("$magnetization$", fontsize = 15)
    pdf.savefig()
    plt.show()

plt.close()

with PdfPages('batch_results/binder_cumulant.pdf') as pdf:
    plt.figure()
    for N in N_values:
        results = open("batch_results/"+str(N), 'rb')
        object_results = pickle.load(results)
        binder_cumulant = np.asarray(object_results[2])
        plt.scatter(binder_cumulant[:,0], binder_cumulant[:, 1], c = np.random.rand(3,1))
        results.close()
        
        plt.hold(True)
    plt.xlabel('$betaJ$',fontsize=15)
    plt.xlim([0,1])
    plt.ylabel("$binder_cumulant$", fontsize = 15)
    pdf.savefig()
    plt.show()

plt.close()

with PdfPages('batch_results/susceptibility.pdf') as pdf:
    plt.figure()
    for N in N_values:
        results = open("batch_results/"+str(N), 'rb')
        object_results = pickle.load(results)
        susceptibility = np.asarray(object_results[1])
        plt.scatter(susceptibility[:, 0], susceptibility[:, 1], c = np.random.rand(3,1))
        results.close()
        
        plt.hold(True)
    plt.xlabel('$betaJ$',fontsize=15)
    plt.xlim([0,1])
    plt.ylabel("$susceptibility$", fontsize = 15)
    pdf.savefig()
    plt.show()

plt.close()

with PdfPages('batch_results/cv.pdf') as pdf:
    plt.figure()
    for N in N_values:
        results = open("batch_results/"+str(N), 'rb')
        object_results = pickle.load(results)
        cv = np.asarray(object_results[3])
        plt.scatter(cv[:,0], cv[:, 1], c = np.random.rand(3,1))
        results.close()
        
        plt.hold(True)
    plt.xlabel('$betaJ$',fontsize=15)
    plt.xlim([0,1])
    plt.ylabel("$cv$", fontsize = 15)
    pdf.savefig()
    plt.show()

plt.close()