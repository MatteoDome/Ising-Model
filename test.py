from simulate_hk import find_links, find_clusters
import numpy as np

N=5
a = np.random.choice([1, -1], size = [N, N])
print(a)
links = find_links(N, a, 1000)
clusters, list_of_labels, _ = find_clusters(N, a, links)

#   Correcting to have the same labels
for i in range(N):
    for j in range(N):
        label = list_of_labels[clusters[i,j]]
        clusters[i, j] = label

print("-----------------")
print(a)
print(clusters)

