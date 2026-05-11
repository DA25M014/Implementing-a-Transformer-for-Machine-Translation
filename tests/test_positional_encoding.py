"""
Unit tests for PositionalEncoding.
Run from repo root:  python tests/test_positional_encoding.py
Mirrors the 5 autograder criteria for the [10M] PE grading block.
"""
import math
import sys
import torch

sys.path.insert(0, ".")
from model import PositionalEncoding


def test_output_shape_preserved():
    pe = PositionalEncoding(d_model=512, dropout=0.0, max_len=200)
    x = torch.randn(8, 50, 512)
    out = pe(x)
    assert out.shape == x.shape, f"shape mismatch {out.shape} vs {x.shape}"
    return "shape preserved"


def test_even_dims_zero_at_pos_zero():
    pe = PositionalEncoding(d_model=512, dropout=0.0, max_len=200)
    out = pe(torch.zeros(1, 1, 512))
    even_vals = out[0, 0, 0::2]
    assert torch.allclose(even_vals, torch.zeros_like(even_vals), atol=1e-6), \
        f"even dims at pos 0 should be 0, max abs = {even_vals.abs().max().item()}"
    return "even dims = sin(0) = 0 at pos 0"


def test_odd_dims_one_at_pos_zero():
    pe = PositionalEncoding(d_model=512, dropout=0.0, max_len=200)
    out = pe(torch.zeros(1, 1, 512))
    odd_vals = out[0, 0, 1::2]
    assert torch.allclose(odd_vals, torch.ones_like(odd_vals), atol=1e-6), \
        f"odd dims at pos 0 should be 1, max deviation = {(odd_vals - 1).abs().max().item()}"
    return "odd dims = cos(0) = 1 at pos 0"


def test_formula_correctness_at_arbitrary_point():
    pe = PositionalEncoding(d_model=512, dropout=0.0, max_len=200)
    out = pe(torch.zeros(1, 200, 512))

    pos, i, d_model = 17, 37, 512
    angle = pos / (10000 ** (2 * i / d_model))

    val_even = out[0, pos, 2 * i].item()
    val_odd  = out[0, pos, 2 * i + 1].item()

    assert math.isclose(val_even, math.sin(angle), abs_tol=1e-5), \
        f"PE[{pos},{2*i}]={val_even}, expected sin={math.sin(angle)}"
    assert math.isclose(val_odd, math.cos(angle), abs_tol=1e-5), \
        f"PE[{pos},{2*i+1}]={val_odd}, expected cos={math.cos(angle)}"
    return "formula correct at (pos=17, i=37)"


def test_encoding_is_buffer_not_parameter():
    pe = PositionalEncoding(d_model=512, dropout=0.0, max_len=200)

    param_names = [n for n, _ in pe.named_parameters()]
    assert "pe" not in param_names, f"PE wrongly registered as parameter: {param_names}"

    buffer_names = [n for n, _ in pe.named_buffers()]
    assert "pe" in buffer_names, f"buffer 'pe' missing; have {buffer_names}"
    return "registered as buffer, not parameter"


if __name__ == "__main__":
    tests = [
        test_output_shape_preserved,
        test_even_dims_zero_at_pos_zero,
        test_odd_dims_one_at_pos_zero,
        test_formula_correctness_at_arbitrary_point,
        test_encoding_is_buffer_not_parameter,
    ]
    print(f"Running {len(tests)} PE tests\n" + "-" * 48)
    failed = 0
    for t in tests:
        try:
            print(f"  PASS  {t():<46}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}");  failed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}");  failed += 1
    print("-" * 48)
    msg = "all passed" if failed == 0 else f"{failed} failed"
    print(f"  {len(tests) - failed}/{len(tests)}  {msg}")
    sys.exit(0 if failed == 0 else 1)
