from ruvector_cluster import cluster
result = cluster([[1, 0], [1.1, 0], [10, 10], [10.1, 10]], 2)
assert result.labels[0] == result.labels[1]
assert result.labels[2] == result.labels[3]
assert result.labels[0] != result.labels[2]
assert result.sizes.sum() == 4
