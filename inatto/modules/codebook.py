"""Codebook — shared, FACE-style (spec §3.3.2).

The frozen MiniLM token embedding table is filtered to "clean English
words" using the COCA-60000 frequency-tagged dictionary intersected
with single-piece BERT/MiniLM tokens (no `##` subwords, only `[a-z]+`).
The filtered token embeddings form a frozen base; a trainable
projection W_c maps them into R^{d_aspect}.

    C[v] = W_c * E_LLM[token_id_v]   for v in filtered vocab

Returns:
    - codebook_raw    Tensor [V, d_llm]   frozen buffer
    - codebook(self)  Tensor [V, d_aspect]  recomputed each forward (W_c is trained)
    - vocabulary      list[str]            the actual words (for descriptor generation)

Used by RQ for codeword selection and by Alignment for descriptor
text generation.
"""

from __future__ import annotations
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Vocabulary filtering
# ---------------------------------------------------------------------------

def _load_coca60000(coca_path: Path, max_words: int = 20000,
                    pos_keep=("N", "J")) -> list[str]:
    """Load the COCA-60000 frequency list and return the top `max_words` open-class words."""
    df = pd.read_excel(coca_path)
    df = df[["PoS", "word"]]
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[df["word"].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    df["word"] = df["word"].apply(
        lambda x: x[1:-1] if x.startswith("(") and x.endswith(")") else x
    )
    df = df.drop_duplicates(subset=["word"], keep="first")
    df = df[df["PoS"].isin(pos_keep)]
    df = df.iloc[:max_words]
    return df["word"].tolist()


def filter_minilm_vocabulary(tokenizer, coca_path: Path | None = None) -> pd.DataFrame:
    """Returns a DataFrame with columns (token, token_id), one row per
    filtered codeword. Filtering rules (same as FACE):
        - single-piece, no `##` subwords
        - lowercase, fullmatch `^[a-z]+$`
        - if coca_path is given, intersect with COCA-60000 open-class words

    The result is sorted by token_id ascending (canonical id ordering).
    """
    vocab = tokenizer.get_vocab()
    rows = [{"token": t, "token_id": i} for t, i in vocab.items()]
    df = pd.DataFrame(rows)
    df = df[~df["token"].str.startswith("##")]
    df["token"] = df["token"].apply(lambda x: tokenizer.convert_tokens_to_string([x]).strip())
    df = df[df["token"].str.fullmatch(r"^[a-z]+$", na=False)]
    if coca_path is not None and coca_path.exists():
        coca_words = set(_load_coca60000(coca_path))
        df = df[df["token"].isin(coca_words)]
    df = df.sort_values(by="token_id").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Codebook module
# ---------------------------------------------------------------------------

class Codebook(nn.Module):
    """Frozen LLM-vocab codebook + trainable projection.

    The codebook is shared between user-side and item-side tokenization
    (LightGCN is bipartite — user and item embeddings live in the same
    256-d space — so a shared codebook is sensible).

    Parameters
    ----------
    llm_path : str | Path
        Path to the MiniLM directory.
    coca_path : str | Path | None
        Optional COCA-60000 xlsx for clean vocab filtering. If None, the
        full single-piece English-word subset of MiniLM is used.
    d_aspect : int
        Output codeword dim (projected from MiniLM 384-d via W_c).
    """

    def __init__(
        self,
        llm_path: str | Path,
        d_aspect: int,
        coca_path: str | Path | None = None,
    ):
        super().__init__()
        llm_path = Path(llm_path)
        tokenizer = AutoTokenizer.from_pretrained(str(llm_path))
        df = filter_minilm_vocabulary(
            tokenizer,
            coca_path=Path(coca_path) if coca_path else None,
        )
        self.vocabulary: list[str] = df["token"].tolist()
        token_ids = torch.tensor(df["token_id"].values, dtype=torch.long)
        self.V = len(self.vocabulary)
        if self.V == 0:
            raise RuntimeError(
                "Codebook ended up empty — check tokenizer path / coca path."
            )

        # Pull the input embedding rows from the LLM once, then drop the model.
        model = AutoModel.from_pretrained(str(llm_path))
        with torch.no_grad():
            E = model.get_input_embeddings()(token_ids).detach().clone()
        self.d_llm = int(E.shape[1])
        self.d_aspect = int(d_aspect)

        self.register_buffer("codebook_raw", E)               # (V, d_llm)
        self.register_buffer("token_ids", token_ids)
        self.W_c = nn.Linear(self.d_llm, self.d_aspect, bias=True)
        # Stash a reference so callers (alignment, descriptor) can reuse
        # the same MiniLM model without reloading.
        self._llm_path = str(llm_path)

    def codebook(self) -> torch.Tensor:
        """Return the current projected codebook (V, d_aspect). W_c is trained."""
        return self.W_c(self.codebook_raw)

    def codebook_normalized(self) -> torch.Tensor:
        C = self.codebook()
        return C / C.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)

    def explain(self, indices) -> list[str]:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return [self.vocabulary[i] for i in indices]

    def reverse_W_c(self, x: torch.Tensor) -> torch.Tensor:
        """Pseudo-inverse mapping d_aspect -> d_llm (for descriptor encoding).

        Forward W_c is a linear map; we invert via Moore-Penrose pseudoinverse.
        """
        W = self.W_c.weight.detach()
        b = self.W_c.bias.detach() if self.W_c.bias is not None else None
        W_pinv = torch.linalg.pinv(W).t()       # (d_aspect, d_llm).T  = (d_llm, d_aspect)
        x_centered = x - b if b is not None else x
        return x_centered @ W_pinv
