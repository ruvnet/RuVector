from ruvector_rerank import rank
hits = rank([1, 0], [[0, 1], [1, 0], [-1, 0]], top_k=2)
assert [hit.index for hit in hits] == [1, 0]
