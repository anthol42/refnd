from refnd import KernelVariant, exact_nearest_neighbors

queries = ["MKTAYIAK"]
refs    = ["MKTAYIAKQR", "ACDEFGHIKLM", "MKTAYIAKQRQ"]
results = exact_nearest_neighbors(
    KernelVariant.ProteinGlobal, queries, refs, k=2
)
print(results[0]) # [(0, 0.20), (2, 0.27)]
# results[0] -> [(2, 0.93), (0, 0.85)]