import pickle
import numpy as np

N_values = [256, 128]

for N in N_values:
    results = open(str(N), 'rb')
    object_results = pickle.load(results)
    binder_cumulant = list(object_results[2])
    with open(str(N) + '_results.txt', 'w') as file:
        for index, result in enumerate(binder_cumulant):
            file.write(str(result[0]) + ' ' + str(result[1]) + '\n')


filenames = ['128_results.txt', '256_results.txt']

for filename in filenames:
    with open(filename, 'r') as unsorted_file:
        values = []
        for line in unsorted_file:
            values.append((float(line.split()[0]), float(line.split()[1])))

    values = sorted(values)


    with open(filename, 'w') as unsorted_file:
        for value in values:
            unsorted_file.write(str(value[0]) + ' ' + str(value[1]) + '\n')
