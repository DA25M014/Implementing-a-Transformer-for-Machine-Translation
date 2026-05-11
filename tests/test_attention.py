"""
Unit tests for scaled_dot_product_attention, MultiHeadAttention, and mask helpers.
Run from repo root:  python tests/test_attention.py
Mirrors the 5 autograder criteria for the [10M] MHA grading block.
"""
import sys
import torch

sys.path.insert(0, ".")
from model import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    make_src_mask,
    make_tgt_mask,
)


def test_sdpa_output_shape():
    B, h, Lq, Lk, d_k, d_v = 2, 8, 7, 11, 64, 64
    Q = torch.randn(B, h, Lq, d_k)
    K = torch.randn(B, h, Lk, d_k)
    V = torch.randn(B, h, Lk, d_v)
    out, alpha = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, h, Lq, d_v), f"output shape {out.shape}"
    assert alpha.shape == (B, h, Lq, Lk), f"attn shape {alpha.shape}"
    return "SDPA output and attention shapes correct"


def test_attention_weights_sum_to_one():
    Q = torch.randn(2, 8, 5, 64)
    K = torch.randn(2, 8, 9, 64)
    V = torch.randn(2, 8, 9, 64)
    _, alpha = scaled_dot_product_attention(Q, K, V)
    sums = alpha.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"weights do not sum to 1; max deviation {(sums - 1).abs().max().item()}"
    return "attention weights sum to 1 over key dim"


def test_masked_positions_get_zero_weight():
    B, h, Lq, Lk, d_k = 2, 4, 3, 6, 16
    Q = torch.randn(B, h, Lq, d_k)
    K = torch.randn(B, h, Lk, d_k)
    V = torch.randn(B, h, Lk, d_k)

    # Mask out the last 2 key positions for the entire batch.
    mask = torch.zeros(B, 1, 1, Lk, dtype=torch.bool)
    mask[:, :, :, -2:] = True

    _, alpha = scaled_dot_product_attention(Q, K, V, mask=mask)
    masked_slice = alpha[..., -2:]
    assert torch.allclose(masked_slice, torch.zeros_like(masked_slice), atol=1e-7), \
        f"masked positions got nonzero weight, max = {masked_slice.abs().max().item()}"

    unmasked_sum = alpha[..., :-2].sum(dim=-1)
    assert torch.allclose(unmasked_sum, torch.ones_like(unmasked_sum), atol=1e-5), \
        f"unmasked weights do not renormalize to 1"
    return "masked positions receive zero attention weight"


def test_mha_output_shape_varying_config():
    cases = [
        (1,  8, 128,  4),
        (4,  4, 256,  8),
        (2, 16, 512, 16),
        (3, 32, 192,  6),
    ]
    for B, L, d_model, h in cases:
        mha = MultiHeadAttention(d_model=d_model, num_heads=h, dropout=0.0)
        x = torch.randn(B, L, d_model)
        out = mha(x, x, x)
        assert out.shape == (B, L, d_model), \
            f"d_model={d_model} h={h}: got {out.shape}"
    return f"MHA shape correct across {len(cases)} (d_model,h) configs"


def test_causal_mask_changes_output():
    torch.manual_seed(0)
    B, L, d_model, h = 2, 10, 64, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=h, dropout=0.0).eval()
    x = torch.randn(B, L, d_model)

    out_unmasked = mha(x, x, x, mask=None)

    causal = torch.ones(L, L, dtype=torch.bool).triu(1).view(1, 1, L, L)
    out_masked = mha(x, x, x, mask=causal)

    diff = (out_unmasked - out_masked).abs().max().item()
    assert diff > 1e-4, f"causal mask did not change output (max diff {diff})"
    return f"causal mask produces different output (max diff {diff:.4f})"


def test_make_src_mask_basic():
    src = torch.tensor([[5, 6, 7, 1, 1], [3, 1, 1, 1, 1]])  # pad_idx=1
    m = make_src_mask(src, pad_idx=1)
    assert m.shape == (2, 1, 1, 5), f"shape {m.shape}"
    assert m.dtype == torch.bool, f"dtype {m.dtype}"
    expected = torch.tensor([[[[False, False, False, True, True]]],
                              [[[False, True,  True,  True, True]]]])
    assert torch.equal(m, expected), "src mask values wrong"
    return "make_src_mask correct"


def test_make_tgt_mask_basic():
    tgt = torch.tensor([[2, 5, 6, 1]])  # pad_idx=1, L=4
    m = make_tgt_mask(tgt, pad_idx=1)
    assert m.shape == (1, 1, 4, 4), f"shape {m.shape}"
    assert m.dtype == torch.bool

    # Row 0 should mask positions 1,2,3 (causal) AND any pads (col 3).
    # Row 3 corresponds to a pad query; the column-pad union still masks col 3.
    # Spot-check: position (row=0, col=0) must be False (attend to self).
    assert m[0, 0, 0, 0].item() is False
    # Position (row=0, col=1) must be True (future).
    assert m[0, 0, 0, 1].item() is True
    # Position (row=2, col=3) must be True (col 3 is pad).
    assert m[0, 0, 2, 3].item() is True
    # Position (row=2, col=2) must be False (current, not pad).
    assert m[0, 0, 2, 2].item() is False
    return "make_tgt_mask correct"


if __name__ == "__main__":
    tests = [
        test_sdpa_output_shape,
        test_attention_weights_sum_to_one,
        test_masked_positions_get_zero_weight,
        test_mha_output_shape_varying_config,
        test_causal_mask_changes_output,
        test_make_src_mask_basic,
        test_make_tgt_mask_basic,
    ]
    print(f"Running {len(tests)} attention tests\n" + "-" * 60)
    failed = 0
    for t in tests:
        try:
            print(f"  PASS  {t():<58}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}");  failed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}");  failed += 1
    print("-" * 60)
    msg = "all passed" if failed == 0 else f"{failed} failed"
    print(f"  {len(tests) - failed}/{len(tests)}  {msg}")
    sys.exit(0 if failed == 0 else 1)
