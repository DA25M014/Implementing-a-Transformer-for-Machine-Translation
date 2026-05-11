"""
Integration smoke test for the full Transformer.
Exercises forward, encode, decode, mask-flow, and weight-tying.
"""
import sys
import torch

sys.path.insert(0, ".")
from model import Transformer, make_src_mask, make_tgt_mask


def test_instantiation_with_no_args():
    """Autograder requirement: Transformer() with no args must succeed."""
    model = Transformer()
    assert isinstance(model, Transformer)
    return "Transformer() with no args instantiates"


def test_forward_shape():
    model = Transformer(
        src_vocab_size=100, tgt_vocab_size=80,
        d_model=64, N=2, num_heads=4, d_ff=128, dropout=0.0,
    ).eval()

    src = torch.randint(2, 100, (2, 5))  # avoid pad_idx=1
    tgt = torch.randint(2, 80,  (2, 4))
    src_mask = make_src_mask(src, pad_idx=1)
    tgt_mask = make_tgt_mask(tgt, pad_idx=1)

    logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (2, 4, 80), f"logits shape {logits.shape}"
    return "forward returns [B, tgt_len, tgt_vocab_size]"


def test_encode_decode_separately():
    model = Transformer(
        src_vocab_size=100, tgt_vocab_size=80,
        d_model=64, N=2, num_heads=4, d_ff=128, dropout=0.0,
    ).eval()

    src = torch.randint(2, 100, (2, 5))
    tgt = torch.randint(2, 80,  (2, 4))
    src_mask = make_src_mask(src, pad_idx=1)
    tgt_mask = make_tgt_mask(tgt, pad_idx=1)

    memory = model.encode(src, src_mask)
    assert memory.shape == (2, 5, 64), f"memory shape {memory.shape}"

    logits = model.decode(memory, src_mask, tgt, tgt_mask)
    assert logits.shape == (2, 4, 80), f"logits shape {logits.shape}"

    # encode + decode should equal forward
    direct = model(src, tgt, src_mask, tgt_mask)
    assert torch.allclose(direct, logits, atol=1e-5), "forward != encode+decode"
    return "encode/decode/forward are consistent"


def test_weight_tying():
    model = Transformer(
        src_vocab_size=50, tgt_vocab_size=40,
        d_model=32, N=1, num_heads=4, d_ff=64,
    )
    assert model.generator.weight.data_ptr() == model.tgt_embed.weight.data_ptr(), \
        "generator weight is not tied to tgt_embed"
    return "decoder embedding tied to output projection"


def test_pad_embedding_zero():
    model = Transformer(
        src_vocab_size=50, tgt_vocab_size=40,
        d_model=32, N=1, num_heads=4, d_ff=64,
    )
    # pad_idx=1 row of both embeddings should be all zeros after init.
    assert torch.allclose(model.src_embed.weight[1], torch.zeros(32))
    assert torch.allclose(model.tgt_embed.weight[1], torch.zeros(32))
    return "pad row in both embeddings is zero-initialised"


def test_param_count_reasonable():
    """Sanity check on total params for a tiny config -- catches duplicated layers."""
    model = Transformer(
        src_vocab_size=100, tgt_vocab_size=100,
        d_model=64, N=2, num_heads=4, d_ff=128,
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # With tied embeddings, expect ~100-200K params for this size.
    assert 50_000 < total < 500_000, f"param count {total} outside sanity range"
    return f"param count {total:,} within sanity range"


def test_loss_decreases_one_step():
    """A single optimizer step should reduce loss on a tiny synthetic batch."""
    torch.manual_seed(25014)
    model = Transformer(
        src_vocab_size=50, tgt_vocab_size=40,
        d_model=32, N=2, num_heads=4, d_ff=64, dropout=0.0,
    )

    src = torch.randint(2, 50, (4, 6))
    tgt = torch.randint(2, 40, (4, 5))
    src_mask = make_src_mask(src, pad_idx=1)
    tgt_mask = make_tgt_mask(tgt, pad_idx=1)
    target_out = torch.randint(2, 40, (4, 5))

    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    logits = model(src, tgt, src_mask, tgt_mask)
    loss_before = crit(logits.reshape(-1, 40), target_out.reshape(-1)).item()

    opt.zero_grad()
    loss = crit(logits.reshape(-1, 40), target_out.reshape(-1))
    loss.backward()
    opt.step()

    logits2 = model(src, tgt, src_mask, tgt_mask)
    loss_after = crit(logits2.reshape(-1, 40), target_out.reshape(-1)).item()

    assert loss_after < loss_before, f"loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"
    return f"loss decreased after one step: {loss_before:.4f} -> {loss_after:.4f}"


if __name__ == "__main__":
    tests = [
        test_instantiation_with_no_args,
        test_forward_shape,
        test_encode_decode_separately,
        test_weight_tying,
        test_pad_embedding_zero,
        test_param_count_reasonable,
        test_loss_decreases_one_step,
    ]
    print(f"Running {len(tests)} Transformer integration tests\n" + "-" * 64)
    failed = 0
    for t in tests:
        try:
            print(f"  PASS  {t():<62}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}");  failed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}");  failed += 1
    print("-" * 64)
    msg = "all passed" if failed == 0 else f"{failed} failed"
    print(f"  {len(tests) - failed}/{len(tests)}  {msg}")
    sys.exit(0 if failed == 0 else 1)
