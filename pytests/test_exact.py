from refnd.core import exact_edges, exact_nearest_neighbors
from refnd.kernels import KernelVariant

DATA = [
    "ACDEFGHIKLMNPQRSTVWY",
    "ACDAFGHKILMNPQRSTVWY",
    "ACDEFGHIPLMNPQRSTVWY",
    "ACDEFGHIKLMIJKPPWY",
    "GGLLPLPKLMNKKKSTVGG",
]


def test_exact_edges():
    edges = exact_edges(KernelVariant.ProteinGlobal, DATA, 0.5)
    edges_no_w = [(i, j) for i, j, w in edges]
    assert edges_no_w == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 3)
    ]


def test_exact_nearest_neighbors():
    knn = exact_nearest_neighbors(KernelVariant.ProteinGlobal, DATA[-2:], DATA[:-2], 2)
    knn = [[i for i, d in neighbors] for neighbors in knn]
    assert knn == [[0, 2], [0, 2]]



if __name__ == "__main__":
    test_exact_edges()
    print("Exact edges OK")
    test_exact_nearest_neighbors()
    print("exact nearest neighbors OK")
