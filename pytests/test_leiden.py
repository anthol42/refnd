from refnd.core import EdgeStore, CsrGraph, INWeightType

EDGES = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)]


def test_subgraph_reindexes_and_maps_nodes():
    store = EdgeStore(4, EDGES)
    g = CsrGraph(store, inweight_type=INWeightType.Similarity)

    sub, mapping = g.subgraph([2, 0, 3])

    assert sub.n == 3
    assert mapping == {2: 0, 0: 1, 3: 2}


def test_subgraph_keeps_only_induced_edges():
    store = EdgeStore(4, EDGES)
    g = CsrGraph(store, inweight_type=INWeightType.Similarity)

    # Node 1 is excluded, so only the (2,3) and (0,3) edges survive.
    sub, mapping = g.subgraph([0, 2, 3])

    new_2, new_0, new_3 = mapping[2], mapping[0], mapping[3]
    assert sorted(n for n, _ in sub.neighbors(new_0)) == [new_3]
    assert sorted(n for n, _ in sub.neighbors(new_2)) == [new_3]
    assert sorted(sub.neighbors(new_3)) == sorted([(new_0, 1.0), (new_2, 1.0)])


if __name__ == "__main__":
    test_subgraph_reindexes_and_maps_nodes()
    print("subgraph reindexes and maps nodes OK")
    test_subgraph_keeps_only_induced_edges()
    print("subgraph keeps only induced edges OK")
