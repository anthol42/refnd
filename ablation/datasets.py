from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from refnd.kernels import KernelVariant
from refnd.kernels.alignments import ScoringMatrix

from cache import CacheStore


@dataclass
class DatasetConfig:
    modality: KernelVariant
    metric: str          # "pcc" | "mcc"
    encoder: str         # HuggingFace model id
    proximity_threshold: float
    kernel_params: dict  # forwarded to kernel constructor as **kwargs


DATASETS: dict[str, DatasetConfig] = {
    "dbaasp": DatasetConfig(
        modality=KernelVariant.AlignmentGlobal,
        metric="pcc",
        encoder="EvolutionaryScale/esmc-300m",
        proximity_threshold=0.5,
        kernel_params={"matrix": ScoringMatrix.Blosum62},
    ),
    "ld50_zhu": DatasetConfig(
        modality=KernelVariant.TanimotoBit,
        metric="pcc",
        encoder="seyonec/ChemBERTa-zinc-base-v1",
        proximity_threshold=0.4,
        kernel_params={},
    ),
    "prom_core_all": DatasetConfig(
        modality=KernelVariant.AlignmentGlobal,
        metric="mcc",
        encoder="zhihan1996/DNABERT-2-117M",
        proximity_threshold=0.45,
        kernel_params={"matrix": ScoringMatrix.Identity},
    ),
}


def load_dataset(name: str, cache: CacheStore) -> tuple[list[Any], np.ndarray]:
    cached = cache.get_dataset(name)
    if cached is not None:
        data, labels = cached
        if name == "ld50_zhu":
            from refnd.utils import BitFingerprint
            data = [BitFingerprint.from_np(row) for row in data]
        return data, labels

    if name == "dbaasp":
        data, labels = _load_dbaasp()
    elif name == "ld50_zhu":
        fp_arrays, labels, smiles = _load_ld50_zhu()
        cache.store_dataset(name, fp_arrays, labels)
        cache.store_dataset(f"{name}_smiles", smiles, np.array([]))
        from refnd.utils import BitFingerprint
        return [BitFingerprint.from_np(row) for row in fp_arrays], labels
    elif name == "prom_core_all":
        data, labels = _load_prom_core_all()
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    cache.store_dataset(name, data, labels)
    return data, labels  # type: ignore[return-value]


def _load_dbaasp() -> tuple[list[str], np.ndarray]:
    from qmap import DBAASPDataset
    import pandas as pd

    print("Downloading DBAASP from HuggingFace...")
    ds = (
        DBAASPDataset()
        .with_l_aa_only()
        .with_canonical_only()
    )
    df = ds.tabular(["sequence", "Escherichia coli"])
    df = df.dropna(subset=["Escherichia coli"])
    # MIC in µg/mL — log-transform for regression
    sequences = df["sequence"].tolist()
    labels = np.log10(df["Escherichia coli"].astype(float).values)
    return sequences, labels


def _load_ld50_zhu() -> tuple[Any, np.ndarray, list[str]]:
    import io
    import requests
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from refnd.utils import BitFingerprint

    url = "https://huggingface.co/datasets/scikit-fingerprints/TDC_ld50_zhu/resolve/main/tdc_ld50_zhu.csv"
    print("Downloading LD50 (Zhu) dataset...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    import pandas as pd
    df = pd.read_csv(io.StringIO(resp.text))
    fp_arrays, labels, smiles = [], [], []
    for _, row in df.iterrows():
        y = float(row["Y"])
        if y <= 0:
            continue
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_arrays.append(np.array(rdkit_fp, dtype=bool))
        labels.append(y)
        smiles.append(row["SMILES"])

    # Store as numpy array so it's picklable; reconstruct BitFingerprint on load
    return np.array(fp_arrays, dtype=bool), np.log10(np.array(labels, dtype=np.float32)), smiles


def _load_prom_core_all() -> tuple[list[str], np.ndarray]:
    import io
    import requests
    import pandas as pd

    url = "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/prom_core_all/train.csv"
    print("Downloading GUE prom_core_all dataset...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    return df["sequence"].tolist(), np.array(df["label"].values, dtype=np.int64)
