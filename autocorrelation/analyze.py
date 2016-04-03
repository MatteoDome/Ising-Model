import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('magnetization_hk.dat', 'rb') as dump:
    data = pickle.load(dump)
    critical = data[0.44000000000000006]-np.mean(data[0.44000000000000006])
    corr = np.correlate(critical, critical, 'full')
    corr = corr[corr.size/2:]
    print(critical)
    plt.plot(np.arange(critical.size), critical)
    # plt.plot(np.arange(corr[:20].size), corr[:20])
    plt.show()

with open('magnetization_metropolis.dat', 'rb') as dump:
    data = pickle.load(dump)
    critical = data[0.44000000000000006]
    corr = np.correlate(critical, critical, 'full')
    corr = corr[corr.size/2:]
    plt.plot(np.arange(critical.size), abs(critical))
    # plt.plot(np.arange(corr[:20].size), corr[:20])
    plt.show()

