"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Boolean mask convention enforced by the skeleton:
    True  -> position is masked out (gets -inf before softmax).
    False -> attend normally.

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : broadcastable to (..., seq_q, seq_k), bool

    Returns:
        output  : (..., seq_q, d_v)
        attn_w  : (..., seq_q, seq_k)  post-softmax weights
    """
    d_k = Q.size(-1)
    scale = d_k ** -0.5

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    alpha = torch.softmax(scores, dim=-1)
    output = torch.matmul(alpha, V)
    return output, alpha


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Encoder padding mask. True at <pad> positions, False elsewhere.

    Shape: [B, 1, 1, src_len] broadcasts over heads and queries
    so one mask covers all (head, query) attention rows.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Decoder mask = padding mask OR causal (look-ahead) mask.

    True at positions to mask out (PAD or future tokens).
    Shape: [B, 1, tgt_len, tgt_len].
    """
    B, L = tgt.shape

    pad_mask = (tgt == pad_idx).view(B, 1, 1, L)

    causal = torch.ones(L, L, dtype=torch.bool, device=tgt.device).triu(1)
    causal = causal.view(1, 1, L, L)

    return pad_mask | causal


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al. 2017, Section 3.2.2).

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        head_i              = Attention(Q W_Q^i, K W_K^i, V W_V^i)

    Implementation notes:
      - Q, K, V are projected by a SINGLE fused linear of width 3*d_model,
        then chunked along the last dim. Fewer parameters in the optimizer
        state and fewer kernel launches than three separate linears.
      - Output projection is named ``out_proj`` so it can be tied to the
        decoder embedding weight externally if desired.
      - The 1/sqrt(d_k) scale is delegated to scaled_dot_product_attention.

    nn.MultiheadAttention is NOT used.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.drop     = nn.Dropout(dropout)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, L, _ = t.shape
        return t.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, h, L, d_k = t.shape
        return t.transpose(1, 2).contiguous().view(B, L, h * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Self-attention is the case query is key is value.
        Cross-attention passes encoder memory as key and value.

        Args:
            query : [B, seq_q, d_model]
            key   : [B, seq_k, d_model]
            value : [B, seq_k, d_model]
            mask  : broadcastable to [B, num_heads, seq_q, seq_k], bool,
                    True = masked out.

        Returns:
            [B, seq_q, d_model]
        """
        if query is key and key is value:
            qkv = self.qkv(query)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            W = self.qkv.weight
            b = self.qkv.bias
            d = self.d_model
            q = torch.nn.functional.linear(query, W[:d],    b[:d])
            k = torch.nn.functional.linear(key,   W[d:2*d], b[d:2*d])
            v = torch.nn.functional.linear(value, W[2*d:],  b[2*d:])

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attended, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        attended = self.drop(attended)

        merged = self._merge_heads(attended)
        return self.out_proj(merged)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017, Section 3.5).

    Even dims carry sine, odd dims carry cosine, with geometrically
    spaced frequencies running from 1 (period 2*pi) down to 1/10000
    (period 2*pi*10000). The table is precomputed once at init and
    stored as a non-persistent buffer -- it is deterministic, so there
    is no reason to bloat checkpoints with ~10 MB of constants.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal PE"

        self.drop = nn.Dropout(dropout)

        table = self._build_sinusoid_table(max_len, d_model)
        self.register_buffer("pe", table, persistent=False)

    @staticmethod
    def _build_sinusoid_table(max_len: int, d_model: int) -> torch.Tensor:
        # Frequencies: omega_i = 10000^(-2i/d_model) for i = 0, 1, ..., d/2 - 1
        half_dim = d_model // 2
        i = torch.arange(half_dim, dtype=torch.float32)
        omega = torch.pow(10000.0, -2.0 * i / d_model)             # [d/2]

        # Positions: 0, 1, ..., max_len - 1
        pos = torch.arange(max_len, dtype=torch.float32)           # [L]

        # Outer product gives angle[p, i] = p * omega_i
        angles = torch.outer(pos, omega)                           # [L, d/2]

        # Interleave even=sin, odd=cos via stack-then-flatten.
        table = torch.stack([angles.sin(), angles.cos()], dim=-1)  # [L, d/2, 2]
        table = table.flatten(start_dim=-2)                        # [L, d_model]

        return table.unsqueeze(0)                                  # [1, L, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -- PE broadcasts over the batch dim.
        L = x.size(1)
        return self.drop(x + self.pe[:, :L])


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network (Vaswani et al. 2017, Section 3.3).

        FFN(x) = max(0, x W1 + b1) W2 + b2

    Two-layer MLP applied independently and identically to each position.
    Dropout is applied after the ReLU activation (paper Section 5.4).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.act     = nn.ReLU()
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, d_model] -> [B, L, d_ff] -> [B, L, d_model]
        return self.linear2(self.drop(self.act(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    One encoder block with PRE-LayerNorm ordering:

        x = x + drop(self_attn(norm1(x)))
        x = x + drop(ffn(norm2(x)))

    Pre-LN is chosen over Post-LN because:
      1. It is stable without aggressive warmup tuning -- gradients of
         attention weights stay well-conditioned from the very first step.
      2. It removes the need to scale residual paths by 1/sqrt(N) for
         deep stacks (Xiong et al. 2020, "On Layer Normalization in
         the Transformer Architecture").
      3. It is the current standard in production transformers (GPT,
         LLaMA, T5-1.1, etc.).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Sub-layer 1: self-attention with pre-norm residual
        normed = self.norm1(x)
        x = x + self.drop(self.self_attn(normed, normed, normed, mask=src_mask))

        # Sub-layer 2: feed-forward with pre-norm residual
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    One decoder block with PRE-LayerNorm ordering:

        x = x + drop(masked_self_attn(norm1(x)))
        x = x + drop(cross_attn(norm2(x), memory, memory))
        x = x + drop(ffn(norm3(x)))

    See EncoderLayer docstring for the Pre-LN justification.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Sub-layer 1: masked self-attention
        normed = self.norm1(x)
        x = x + self.drop(self.self_attn(normed, normed, normed, mask=tgt_mask))

        # Sub-layer 2: cross-attention to encoder memory
        normed = self.norm2(x)
        x = x + self.drop(self.cross_attn(normed, memory, memory, mask=src_mask))

        # Sub-layer 3: feed-forward
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Stack of N identical EncoderLayer modules with a final LayerNorm.

    The final norm is necessary under Pre-LN: without it, the very last
    residual addition leaves activations un-normalized before they reach
    the decoder's cross-attention.

    Layers are constructed fresh (no copy.deepcopy) so each has its own
    initialised parameters and there is no shared-state confusion.
    """

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        # Pull config off the prototype layer; clone by re-construction.
        d_model   = layer.norm1.normalized_shape[0]
        num_heads = layer.self_attn.num_heads
        d_ff      = layer.ffn.linear1.out_features
        dropout   = layer.drop.p

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    """
    Stack of N identical DecoderLayer modules with a final LayerNorm.
    Same Pre-LN motivation as Encoder.
    """

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        d_model   = layer.norm1.normalized_shape[0]
        num_heads = layer.self_attn.num_heads
        d_ff      = layer.ffn.linear1.out_features
        dropout   = layer.drop.p

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer for sequence-to-sequence translation.

    Self-bootstrapping: a bare ``Transformer()`` call loads vocabularies,
    spaCy tokenizers, and (when configured) downloads pretrained weights
    via gdown -- all inside __init__, per the assignment\'s autograder contract.

    Architecture choices (full rationale in STYLE.md):
      - Pre-LayerNorm throughout.
      - Decoder input embedding TIED to output projection (Press & Wolf 2017).
      - Source and target embeddings independent.
      - Embeddings scaled by sqrt(d_model) before adding PE (paper Section 3.4).
      - Xavier-uniform init on Linear, Normal(0, d_model^-0.5) on Embeddings.

    Defaults for vocab sizes match the Multi30k vocab built from train split
    with min_freq=2: de=8012, en=6190. These are baked in so the autograder
    can call ``Transformer()`` with no arguments and get the right architecture.
    """

    # ── Checkpoint plumbing ──────────────────────────────────────────
    # When CHECKPOINT_GDRIVE_ID is non-None, __init__ will download the
    # corresponding .pt file from Google Drive via gdown and load weights.
    # Set this AFTER the first successful Kaggle training run that produces
    # the canonical transformer_main.pt.
    CHECKPOINT_GDRIVE_ID: str | None = "15e-O7Ji4kMnLrmaTBY5hmjnW6xpuJKGS"
    CHECKPOINT_LOCAL_NAME: str       = "transformer_main.pt"

    def __init__(
        self,
        src_vocab_size: int = 8012,
        tgt_vocab_size: int = 6190,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.15,
        max_len:   int   = 128,
        artifacts_dir: str = "artifacts",
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.d_model        = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len        = max_len
        self.artifacts_dir  = artifacts_dir

        # ── Architecture ─────────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout=dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout=dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Tied output projection.
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.generator.weight = self.tgt_embed.weight

        self._init_weights()

        # ── Vocab + tokenizer auto-load ──────────────────────────────
        # Per the announcement: vocab and tokenizers must load in __init__.
        # Failure here is NON-FATAL during unit tests where artifacts may
        # be absent; only infer() will then raise.
        self.src_vocab     = None
        self.tgt_vocab     = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self._load_vocab_and_tokenizers(artifacts_dir)

        # ── Optional weight download + load ──────────────────────────
        # checkpoint_path argument lets callers force a specific path;
        # otherwise we fall back to CHECKPOINT_GDRIVE_ID + CHECKPOINT_LOCAL_NAME.
        target_path = checkpoint_path or self.CHECKPOINT_LOCAL_NAME
        if self.CHECKPOINT_GDRIVE_ID is not None or checkpoint_path is not None:
            self._maybe_download_and_load_weights(target_path)

    # ── Bootstrapping helpers ────────────────────────────────────────

    def _load_vocab_and_tokenizers(self, artifacts_dir: str) -> None:
        """
        Load vocabs from artifacts/ and spaCy tokenizers eagerly.

        If a spaCy model is missing, attempt a runtime install via
        ``python -m spacy download``. This makes the Transformer
        self-sufficient on autograder machines that did not pre-install
        the language models.

        Imports are local so unit tests of pure-architecture code do not
        require the dataset module / spaCy to be importable.
        """
        try:
            from pathlib import Path as _P
            from dataset import Vocab, _make_spacy_tokenizer

            adir = _P(artifacts_dir)
            de_path = adir / "vocab_de.pt"
            en_path = adir / "vocab_en.pt"
            if de_path.exists() and en_path.exists():
                self.src_vocab = Vocab.load(de_path)
                self.tgt_vocab = Vocab.load(en_path)

            self.src_tokenizer = self._safe_load_spacy("de_core_news_sm")
            self.tgt_tokenizer = self._safe_load_spacy("en_core_web_sm")
        except Exception as e:
            import warnings
            warnings.warn(f"Transformer bootstrap (vocab/tokenizer) skipped: {e}")

    @staticmethod
    def _safe_load_spacy(model_name: str):
        """
        Return a callable str->list[str] tokenizer for the given spaCy model.
        If the model is not installed, run ``python -m spacy download`` first.
        """
        import sys, subprocess, spacy
        from dataset import _make_spacy_tokenizer
        try:
            return _make_spacy_tokenizer(model_name)
        except OSError:
            # Model not installed -- fetch it once, then retry.
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
            )
            # spaCy sometimes needs a fresh import after install.
            import importlib
            importlib.invalidate_caches()
            return _make_spacy_tokenizer(model_name)

    def _maybe_download_and_load_weights(self, path: str) -> None:
        """Download via gdown if not present, then load_state_dict."""
        import os
        if not os.path.exists(path):
            if self.CHECKPOINT_GDRIVE_ID is None:
                return
            gdown.download(id=self.CHECKPOINT_GDRIVE_ID, output=path, quiet=False)

        if os.path.exists(path):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            # Support both "raw state_dict" and "{model_state_dict: ...}" formats.
            state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing or unexpected:
                import warnings
                warnings.warn(
                    f"load_state_dict had missing={missing} unexpected={unexpected}"
                )

    def _init_weights(self) -> None:
        """Xavier-uniform on Linear; Normal(0, d^-0.5) on Embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.generator:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.d_model ** -0.5)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.src_embed(src) * (self.d_model ** 0.5)
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.tgt_embed(tgt) * (self.d_model ** 0.5)
        y = self.pos_enc(y)
        y = self.decoder(y, memory, src_mask, tgt_mask)
        return self.generator(y)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translate a single German sentence to English via greedy decoding.

        End-to-end pipeline:
          1. spaCy-tokenize the input German string.
          2. Wrap in <sos> ... <eos> and convert to ids via src_vocab.
          3. Build the src padding mask (trivially all-False for one sentence).
          4. Call train.greedy_decode for autoregressive generation.
          5. Strip specials and join tokens with spaces.

        Returns the English translation as a plain string.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Vocab not loaded. Ensure artifacts/vocab_de.pt and vocab_en.pt exist."
            )
        if self.src_tokenizer is None:
            raise RuntimeError(
                "spaCy tokenizer not loaded. Check de_core_news_sm installation."
            )

        # Local import to avoid circular dependency (train.py imports model.py).
        from train import beam_search_decode

        device = next(self.parameters()).device

        # Tokenize and convert to ids (with <sos>/<eos>).
        de_tokens = self.src_tokenizer(src_sentence)[: self.max_len - 2]
        src_ids   = self.src_vocab.encode(de_tokens, add_specials=True)
        src       = torch.tensor([src_ids], dtype=torch.long, device=device)

        # Source padding mask.
        src_mask = (src == self.tgt_vocab.PAD_IDX).unsqueeze(1).unsqueeze(2)

        # Beam=2 search with tight length cap. Per-sentence cost is ~2x greedy
        # which fits comfortably inside autograder timeout budgets.
        out_max_len = min(self.max_len, len(src_ids) + 8)

        ys = beam_search_decode(
            model=self,
            src=src,
            src_mask=src_mask,
            max_len=out_max_len,
            start_symbol=self.tgt_vocab.SOS_IDX,
            end_symbol=self.tgt_vocab.EOS_IDX,
            pad_idx=self.tgt_vocab.PAD_IDX,
            beam_size=2,
            length_penalty=0.6,
            device=str(device),
        )

        # Detokenize: strip specials, join with spaces, then fix punctuation.
        out_ids = ys[0].tolist()
        en_tokens = self.tgt_vocab.decode(out_ids, strip_specials=True)
        text = " ".join(en_tokens)
        return self._detokenize(text)

    @staticmethod
    def _detokenize(text: str) -> str:
        """
        Collapse whitespace introduced by space-joining BPE-style tokens
        back into natural English punctuation. Matches the convention
        sacrebleu\'s default tokenizer expects on reference strings.
        """
        import re as _re
        # Remove space BEFORE: . , ! ? ; : %  and closing brackets ) ] }
        text = _re.sub(r"\s+([.,!?;:%)\]\}])", r"\1", text)
        # Remove space AFTER opening brackets ( [ {  and dollar/hash etc.
        text = _re.sub(r"([(\[\{\$#@])\s+", r"\1", text)
        # Glue contractions:  n't  's  're  've  'll  'd  'm
        text = _re.sub(r"\s+(n\u0027t|\u0027s|\u0027re|\u0027ve|\u0027ll|\u0027d|\u0027m)\b",
                       r"\1", text)
        text = _re.sub(r"\s+('t|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
        # Collapse multiple internal spaces
        text = _re.sub(r"\s{2,}", " ", text)
        return text.strip()
