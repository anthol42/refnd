from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from refnd.kernels import KernelVariant
from refnd.kernels.alignments import ScoringMatrix, CoverageMode, LocalIdentityMode

from cache import CacheStore


@dataclass
class DatasetConfig:
    modality: KernelVariant
    metric: str | None          # "pcc" | "mcc" | None (no supervised task)
    encoder: str | None         # HuggingFace model id | None (no embeddings)
    proximity_threshold: float
    kernel_params: dict         # forwarded to kernel constructor as **kwargs


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
        kernel_params={"matrix": ScoringMatrix.Dnafull,
                       "identity_mode": LocalIdentityMode.MinSeqLength,
                       "cov_mode": CoverageMode.ShorterSeq,
                       "min_coverage": 0.7},
    ),
    "peptide_atlas": DatasetConfig(
        modality=KernelVariant.AlignmentGlobal,
        metric=None,
        encoder=None,
        proximity_threshold=0.5,
        kernel_params={"matrix": ScoringMatrix.Blosum62},
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

    if name == "peptide_atlas":
        data, labels = _load_peptide_atlas()
        cache.store_dataset(name, data, labels)
        return data, labels
    elif name == "dbaasp":
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


def _load_peptide_atlas() -> tuple[list[str], np.ndarray]:
    fasta_path = Path(".cache/peptide_atlas.fasta")
    if not fasta_path.exists():
        import re
        import sys
        from glob import glob
        from warnings import warn
        import requests
        from bs4 import BeautifulSoup
        from tqdm import tqdm

        cache_files = Path(".cache/files")
        cache_files.mkdir(parents=True, exist_ok=True)

        print("Fetching PeptideAtlas build list...")
        resp = requests.get("https://peptideatlas.org/builds/", timeout=60)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "bdtable"})
        if table is None:
            raise ValueError("PeptideAtlas build table not found")

        thead      = table.find("thead")
        header_row = (thead or table).find("tr")
        headers    = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        tbody      = table.find("tbody")
        data_rows  = (tbody or table).find_all("tr")
        if not thead:
            data_rows = data_rows[1:]

        rows = []
        for row in data_rows:
            cells    = row.find_all(["td", "th"])
            row_data = []
            for cell in cells:
                link = cell.find("a")
                if link and link.get("href") and link.get_text(strip=True).endswith(".fasta"):
                    row_data.append(f"[{link.get_text(strip=True)}]({link['href']})")
                else:
                    row_data.append(cell.get_text(strip=True))
            if row_data:
                rows.append(row_data)

        import pandas as pd
        max_cols = len(headers)
        rows     = [r[:max_cols] + [""] * (max_cols - len(r)) for r in rows]
        df       = pd.DataFrame(rows, columns=headers)
        df       = df.loc[df["Build Name"] != ""]

        entries = [s for s in df["Peptide Sequences"].tolist() if s]
        parsed  = [re.findall(r"(\[.*?])(\(.*?\))", s)[0] for s in entries]
        names   = [e[0][1:-1] for e in parsed]
        urls    = [e[1][1:-1] for e in parsed]

        print(f"Downloading {len(names)} FASTA files...")
        for name, url in tqdm(zip(names, urls), total=len(names)):
            local = cache_files / name
            if local.exists():
                continue
            r = requests.get(f"https://peptideatlas.org/builds/{url}", timeout=120)
            if not r.ok:
                warn(Warning(f"Failed to fetch {name} (HTTP {r.status_code})"))
                continue
            local.write_text(r.text)

        print("Merging and deduplicating sequences...")
        all_sequences: set[str] = set()
        for file in glob(str(cache_files / "*.fasta")):
            with open(file) as f:
                current_id = None
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        current_id = line[1:]
                    elif current_id is not None and line:
                        all_sequences.add(line)
        all_sequences = {seq for seq in all_sequences if len(seq) <= 100}
        with open(fasta_path, "w") as f:
            for i, seq in enumerate(all_sequences):
                f.write(f">seq_{i}\n{seq}\n")
        print(f"  Wrote {len(all_sequences):,} sequences to {fasta_path}")

    sequences = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">") and line:
                sequences.append(line)
    return sequences, np.zeros(len(sequences), dtype=np.float32)


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
