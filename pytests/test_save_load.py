import tempfile
from pathlib import Path

from refnd.core import HNSWState
from refnd.kernels import KernelVariant

DATA = [
    "ACDEFGHIKLMNPQRSTVWY",
    "MKTAYIAKQRQISFVKSHFSRQ",
    "GASDFLKJHQWERTYUIOPAS",
    "PEPTIDESEQUENCEFASTA",
    "MNGTEGPNFYVPFSNKTGVV",
    "QWERTYUIOPASDFGHJKLZ",
    "LMNPQRSTVWYACDEFGHIK",
]

QUERIES = [
    "ACDEFGHIKLMNPQRSTVWY",
    "GASDFLKJHQWERTYUIOPAS",
]


def _build_and_search(state):
    state.build(progress=False)
    return state.search(QUERIES, k=3, ef=10)


def test_new_accepts_list_and_generator_identically():
    """new()'s bulk (sized) and iterator (unsized) paths must produce equivalent graphs."""
    from_list = HNSWState(KernelVariant.AlignmentGlobal, DATA)
    from_gen = HNSWState(KernelVariant.AlignmentGlobal, (s for s in DATA))

    results_list = _build_and_search(from_list)
    results_gen = _build_and_search(from_gen)

    assert len(results_list) == len(results_gen) == len(QUERIES)
    for hits_list, hits_gen in zip(results_list, results_gen):
        assert hits_list == hits_gen


def test_save_load_roundtrip_with_list():
    state = HNSWState(KernelVariant.AlignmentGlobal, DATA)
    state.build(progress=False)
    expected = state.search(QUERIES, k=3, ef=10)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.hnsw")
        state.save(path)
        loaded = HNSWState.load(KernelVariant.AlignmentGlobal, path, DATA)
        assert loaded.is_built
        assert loaded.search(QUERIES, k=3, ef=10) == expected


def test_save_load_roundtrip_with_generator():
    """load()'s data argument accepts a generator too (the fallback path added
    alongside new()'s), not just a sized sequence."""
    state = HNSWState(KernelVariant.AlignmentGlobal, DATA)
    state.build(progress=False)
    expected = state.search(QUERIES, k=3, ef=10)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.hnsw")
        state.save(path)
        loaded = HNSWState.load(KernelVariant.AlignmentGlobal, path, (s for s in DATA))
        assert loaded.is_built
        assert loaded.search(QUERIES, k=3, ef=10) == expected


if __name__ == "__main__":
    test_new_accepts_list_and_generator_identically()
    print("new() list/generator equivalence OK")
    test_save_load_roundtrip_with_list()
    print("save/load with list OK")
    test_save_load_roundtrip_with_generator()
    print("save/load with generator OK")
