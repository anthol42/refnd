from refnd.core import HNSWState, HNSWIndex, HNSWConfig
from refnd.kernels import KernelVariant

DATA = [
    "ACDEFGHIKLMNPQRSTVWY",
    "MKTAYIAKQRQISFVKSHFSRQ",
    "GASDFLKJHQWERTYUIOPAS",
    "PEPTIDESEQUENCEFASTA",
    "MNGTEGPNFYVPFSNKTGVV",
]


def test_config_repr():
    state = HNSWState(KernelVariant.ProteinGlobal, DATA)
    config = state.config
    assert isinstance(config, HNSWConfig)
    s = str(config)
    assert "HNSWConfig" in s
    print(s)


def test_index_repr():
    state = HNSWState(KernelVariant.ProteinGlobal, DATA)
    state.build(progress=False)
    idx = state.index
    assert isinstance(idx, HNSWIndex)
    s = str(idx)
    assert "dataset_size" in s
    assert "entry_point" in s
    assert "max_layers" in s
    assert "n_edges" in s
    assert "layers" in s
    assert "config" in s
    print(s)


def test_index_fields():
    state = HNSWState(KernelVariant.ProteinGlobal, DATA)
    state.build(progress=False)
    idx = state.index
    assert idx.dataset_size == len(DATA)
    assert idx.max_layers >= 1
    assert isinstance(idx.layers, list)
    assert isinstance(idx.proximity_edges, list)
    assert isinstance(idx.config, HNSWConfig)


if __name__ == "__main__":
    test_config_repr()
    print("config repr OK")
    test_index_repr()
    print("index repr OK")
    test_index_fields()
    print("index fields OK")
