"""Two-layer MLP with early stopping, supporting PCC (regression) and MCC (classification)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_eval_mlp(
    embs: torch.Tensor,
    labels: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    metric: str,                # "pcc" or "mcc"
    epochs: int = 300,
    lr: float = 0.003,
    patience: int = 15,
    batch_size: int = 64,
    seed: int = 42,
) -> dict:
    """Train MLP on train_idx, use val_idx for early stopping, evaluate on test_idx."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_classification = metric == "mcc"
    n_classes = int(np.max(labels) + 1) if is_classification else 1

    # --- build tensors ---
    X_all = embs.float()
    if is_classification:
        y_all = torch.tensor(labels, dtype=torch.long)
    else:
        y_all = torch.tensor(labels, dtype=torch.float32)

    tr_idx  = np.array(train_idx)

    val_idx_arr = np.array(val_idx)

    X_tr,  y_tr  = X_all[tr_idx].to(device),       y_all[tr_idx].to(device)
    X_val, y_val = X_all[val_idx_arr].to(device),  y_all[val_idx_arr].to(device)
    X_te         = X_all[test_idx].to(device)

    # --- model ---
    in_dim = X_all.shape[1]
    out_dim = n_classes if is_classification else 1
    model = MLP(in_dim, out_dim).to(device)

    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out.squeeze(-1) if not is_classification else out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(
                val_out.squeeze(-1) if not is_classification else val_out, y_val
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- evaluate ---
    model.eval()
    with torch.no_grad():
        test_out = model(X_te).cpu()

    y_te = y_all[test_idx].numpy()

    if is_classification:
        preds = test_out.argmax(dim=-1).numpy()
        score = float(matthews_corrcoef(y_te, preds))
    else:
        preds = test_out.squeeze(-1).numpy()
        score, _ = pearsonr(preds, y_te)
        score = float(score)

    return {"metric": metric, "score": score}
