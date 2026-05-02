from refnd.core import HNSWState
from refnd.kernels import KernelVariant

DATA = [
    "ACDEFGHIKLMNPQRSTVWY",
    "MKTAYIAKQRQISFVKSHFSRQ",
    "GASDFLKJHQWERTYUIOPAS",
    "PEPTIDESEQUENCEFASTA",
    "MNGTEGPNFYVPFSNKTGVV",
]

QUERIES = [
    "ACDEFGHIKLMNPQRSTVWY",
    "GASDFLKJHQWERTYUIOPAS",
]


def test_global():
    index = HNSWState(KernelVariant.ProteinGlobal, DATA)
    index.build(progress=True)
    results = index.search(QUERIES, k=3, ef=10, progress=True)
    assert len(results) == 2
    for hits in results:
        assert len(hits) <= 3
        for idx, dist in hits:
            assert 0 <= idx < len(DATA)
            assert 0.0 <= dist <= 1.0


def test_local():
    index = HNSWState(KernelVariant.ProteinLocal, DATA)
    index.build(progress=False)
    results = index.search(QUERIES, k=2, ef=10)
    assert len(results) == 2


if __name__ == "__main__":
    test_global()
    print("global OK")
    test_local()
    print("local OK")

