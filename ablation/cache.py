import pickle
from pathlib import Path
from typing import Any

import torch

from refnd.core import EdgeStore


class CacheStore:
    def __init__(self, root: str = ".cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- edges ---

    def get_edges(self, name: str) -> EdgeStore | None:
        path = self.root / f"{name}.edgestr"
        if not path.exists():
            return None
        return EdgeStore.load(str(path))

    def store_edges(self, name: str, es: EdgeStore) -> None:
        es.save(str(self.root / f"{name}.edgestr"))

    # --- embeddings ---

    def get_embs(self, name: str) -> torch.Tensor | None:
        path = self.root / f"{name}.pth"
        if not path.exists():
            return None
        return torch.load(str(path), weights_only=True)

    def store_embs(self, name: str, t: torch.Tensor) -> None:
        torch.save(t, str(self.root / f"{name}.pth"))

    # --- raw dataset ---

    def get_dataset(self, name: str) -> tuple[Any, Any] | None:
        path = self.root / f"{name}.pkl"
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def store_dataset(self, name: str, data: Any, labels: Any) -> None:
        with open(self.root / f"{name}.pkl", "wb") as f:
            pickle.dump((data, labels), f)
