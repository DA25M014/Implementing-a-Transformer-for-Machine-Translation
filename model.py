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
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        #   self.linear1 = nn.Linear(d_model, d_ff)
        #   self.linear2 = nn.Linear(d_ff, d_model)
        #   self.dropout = nn.Dropout(p=dropout)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO:instantiate:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: instantiate:
        raise NotImplementedError

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        raise NotImplementedError


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()
        # TODO: Instantiate 
        # init should also load the model weights if checkpoint path provided, download the .pth file like this
        if checkpoint_path is not None:
            gdown.download(id="<.pth drive id>", output=checkpoint_path, quiet=False)
        raise NotImplementedError

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        raise NotImplementedError

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        raise NotImplementedError

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        raise NotImplementedError


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        raise NotImplementedError