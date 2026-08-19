import numpy as np
import pytest

from refnd.core import exact_edges, exact_nearest_neighbors, HNSWState
from refnd.kernels import KernelVariant, zip_kernel
from refnd.kernels.vectors import Cosine, L1, L2
from refnd.utils import Vector


def np_cosine(a, b):
    return 1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def np_l1(a, b):
    return np.sum(np.abs(a - b))


def np_l2(a, b):
    return np.linalg.norm(a - b)


KERNELS = [
    (Cosine, KernelVariant.Cosine, np_cosine),
    (L1, KernelVariant.L1, np_l1),
    (L2, KernelVariant.L2, np_l2),
]

rng = np.random.default_rng(42)
DATA = rng.normal(size=(6, 8)).astype(np.float32)


# ── direct .call() vs numpy ────────────────────────────────────────────────────

@pytest.mark.parametrize("kernel_cls, variant, np_ref", KERNELS)
def test_call_matches_numpy(kernel_cls, variant, np_ref):
    k = kernel_cls()
    for a in DATA:
        for b in DATA:
            got = k.call(a, b)
            expected = np_ref(a.astype(np.float64), b.astype(np.float64))
            assert got == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("kernel_cls, variant, np_ref", KERNELS)
def test_dunder_call_matches_call(kernel_cls, variant, np_ref):
    k = kernel_cls()
    a, b = DATA[0], DATA[1]
    assert k(a, b) == k.call(a, b)


def test_cosine_zero_vector_is_one_not_nan():
    zero = np.zeros(4, dtype=np.float32)
    other = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert Cosine().call(zero, other) == 1.0


# ── input validation ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kernel_cls, variant, np_ref", KERNELS)
def test_call_rejects_float64(kernel_cls, variant, np_ref):
    a = np.array([1.0, 2.0], dtype=np.float64)
    b = np.array([3.0, 4.0], dtype=np.float64)
    with pytest.raises(TypeError):
        kernel_cls().call(a, b)


@pytest.mark.parametrize("kernel_cls, variant, np_ref", KERNELS)
def test_call_rejects_noncontiguous(kernel_cls, variant, np_ref):
    base = np.arange(10, dtype=np.float32)
    strided = base[::2]
    with pytest.raises(TypeError):
        kernel_cls().call(strided, strided)


# ── Vector wrapper ────────────────────────────────────────────────────────────

def test_vector_from_list_matches_from_np():
    values = [1.0, 2.0, 3.0, 4.0]
    v_list = Vector.from_list(values)
    v_np = Vector(np.array(values, dtype=np.float32))
    assert v_list.to_list() == v_np.to_list()
    assert len(v_list) == len(v_np)


def test_vector_to_np_roundtrip():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = Vector(values)
    np.testing.assert_array_equal(v.to_np(), values)


# ── batch APIs vs numpy brute force ─────────────────────────────────────────────

@pytest.mark.parametrize("kernel_cls, variant, np_ref", KERNELS)
def test_exact_edges_matches_numpy_bruteforce(kernel_cls, variant, np_ref):
    threshold = 2.0
    expected = {
        (i, j)
        for i in range(len(DATA))
        for j in range(i + 1, len(DATA))
        if np_ref(DATA[i].astype(np.float64), DATA[j].astype(np.float64)) <= threshold
    }
    store = exact_edges(variant, list(DATA), threshold, progress=False)
    got = {(i, j) for i, j, _ in store}
    assert got == expected


@pytest.mark.parametrize("kernel_cls, variant, np_ref", KERNELS)
def test_exact_nearest_neighbors_matches_numpy_bruteforce(kernel_cls, variant, np_ref):
    queries = DATA[:2]
    refs = DATA[2:]
    k = 2
    results = exact_nearest_neighbors(variant, list(queries), list(refs), k, progress=False)
    for q, hits in zip(queries, results):
        dists = [np_ref(q.astype(np.float64), r.astype(np.float64)) for r in refs]
        expected_order = sorted(range(len(refs)), key=lambda i: dists[i])[:k]
        got_order = [i for i, _ in hits]
        assert got_order == expected_order
        for (i, d) in hits:
            assert d == pytest.approx(dists[i], abs=1e-5)


@pytest.mark.parametrize("kernel_cls, variant, np_ref", KERNELS)
def test_zip_kernel_matches_numpy(kernel_cls, variant, np_ref):
    data1 = DATA[:3]
    data2 = DATA[3:]
    scores = zip_kernel(variant, list(data1), list(data2), progress=False)
    expected = [np_ref(a.astype(np.float64), b.astype(np.float64)) for a, b in zip(data1, data2)]
    for got, exp in zip(scores, expected):
        assert got == pytest.approx(exp, abs=1e-5)


@pytest.mark.parametrize("kernel_cls, variant, np_ref", KERNELS)
def test_hnsw_matches_exact_for_small_dataset(kernel_cls, variant, np_ref):
    # ef large enough that HNSW's approximate search is exact for this tiny dataset.
    state = HNSWState(variant, list(DATA), ef_construction=200)
    state.build(progress=False)
    query = DATA[:1]
    k = 3
    hnsw_hits = state.search(list(query), k=k, ef=200)[0]

    dists = [np_ref(query[0].astype(np.float64), r.astype(np.float64)) for r in DATA]
    expected_order = sorted(range(len(DATA)), key=lambda i: dists[i])[:k]
    got_order = [i for i, _ in hnsw_hits]
    assert got_order == expected_order


if __name__ == "__main__":
    for kernel_cls, variant, np_ref in KERNELS:
        test_call_matches_numpy(kernel_cls, variant, np_ref)
        test_exact_edges_matches_numpy_bruteforce(kernel_cls, variant, np_ref)
        test_exact_nearest_neighbors_matches_numpy_bruteforce(kernel_cls, variant, np_ref)
        test_zip_kernel_matches_numpy(kernel_cls, variant, np_ref)
        test_hnsw_matches_exact_for_small_dataset(kernel_cls, variant, np_ref)
    print("All vector kernel tests OK")
