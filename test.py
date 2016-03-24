from simulate_hk import find_links, find_clusters2
import numpy as np

N=5
a = np.random.choice([1, -1], size = [N, N])
print(a)
links = find_links(N, a, 1000)
clusters, list_of_labels = find_clusters2(N, a, links)

#   Correcting to have the same labels
for i in range(N):
    for j in range(N):
        label = clusters[i, j]
        while(label != list_of_labels[label]):
            label = list_of_labels[label]
        clusters[i, j] = label

print("-----------------")
print(a)
print(clusters)
print(list_of_labels)

