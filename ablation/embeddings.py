"""Foundation model embedding computation with caching."""

from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from cache import CacheStore
from datasets import DatasetConfig

BATCH_SIZE = 64


def compute_embeddings(
    dataset_name: str,
    data: list[Any],
    cfg: DatasetConfig,
    cache: CacheStore,
    device: str | None = None,
) -> torch.Tensor:
    cached = cache.get_embs(dataset_name)
    if cached is not None:
        return cached

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = cfg.encoder
    if "esmc" in encoder.lower() or "esm" in encoder.lower():
        embs = _embed_esmc(data, device)
    elif "chemberta" in encoder.lower() or "chem" in encoder.lower():
        # data is list[BitFingerprint]; ChemBERTa needs SMILES strings
        smiles_entry = cache.get_dataset(f"{dataset_name}_smiles")
        if smiles_entry is None:
            raise RuntimeError(f"SMILES cache missing for {dataset_name}. Re-run load_dataset first.")
        smiles, _ = smiles_entry
        embs = _embed_hf(encoder, smiles, device)
    else:
        embs = _embed_hf(encoder, data, device)

    cache.store_embs(dataset_name, embs)
    return embs


def _embed_esmc(sequences: list[str], device: str) -> torch.Tensor:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein

    model = ESMC.from_pretrained("esmc_300m").to(device).eval()
    pad_id = model.tokenizer.pad_token_id
    eos_id = model.tokenizer.eos_token_id

    all_embs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="ESM-C embeddings"):
            batch = sequences[i : i + BATCH_SIZE]
            tokens_list = [
                model.encode(ESMProtein(sequence=seq)).sequence for seq in batch
            ]
            tokens = torch.nn.utils.rnn.pad_sequence(
                tokens_list, batch_first=True, padding_value=pad_id
            ).to(device)

            mask = (tokens != pad_id) & (tokens != eos_id)
            mask[:, 0] = False  # exclude BOS

            emb = model(tokens).embeddings  # (B, L, D)
            for j in range(len(batch)):
                m = mask[j]
                all_embs.append(emb[j, m, :].mean(dim=0).cpu())

    return torch.stack(all_embs).to(torch.float32)


def _embed_hf(model_name: str, data: list[Any], device: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # transformers==4.57.6 (Biohub fork, pulled in by esm) has a bug where
    # AutoModel.register() rejects remote config classes that are structurally
    # identical to built-in ones but differ by Python module path (e.g. DNABERT-2).
    # Patch the check away temporarily so the correct remote model class is used.
    import transformers.models.auto.auto_factory as _af

    _cls = _af._BaseAutoModelClass
    _orig_register = _cls.register

    @classmethod  # type: ignore[misc]
    def _patched_register(cls, config_class, model_class, exist_ok=False):
        model_class.config_class = config_class
        _orig_register.__func__(cls, config_class, model_class, exist_ok=True)

    _cls.register = _patched_register
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    finally:
        _cls.register = _orig_register

    texts = [str(x) for x in data]

    all_embs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{model_name} embeddings"):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            out = model(**enc)
            # Some models (e.g. DNABERT-2) return a tuple rather than ModelOutput
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            # mean pooling over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0).to(torch.float32)
