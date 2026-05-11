"""
Smoke tests for FFN, EncoderLayer, DecoderLayer, Encoder, Decoder.
These have no autograder unit tests of their own -- they are exercised
end-to-end by training. The point here is to catch shape and wiring
bugs early before they explode inside a training loop.
"""
import sys
import torch

sys.path.insert(0, ".")
from model import (
    PositionwiseFeedForward,
    EncoderLayer, DecoderLayer,
    Encoder, Decoder,
    make_src_mask, make_tgt_mask,
)


def test_ffn_shape():
    ffn = PositionwiseFeedForward(d_model=128, d_ff=512, dropout=0.0)
    x = torch.randn(2, 9, 128)
    out = ffn(x)
    assert out.shape == x.shape, f"FFN shape {out.shape}"
    return "FFN preserves [B, L, d_model] shape"


def test_encoder_layer_shape():
    layer = EncoderLayer(d_model=128, num_heads=4, d_ff=512, dropout=0.0)
    x = torch.randn(2, 9, 128)
    mask = torch.zeros(2, 1, 1, 9, dtype=torch.bool)
    out = layer(x, mask)
    assert out.shape == x.shape, f"EncoderLayer shape {out.shape}"
    return "EncoderLayer preserves shape"


def test_decoder_layer_shape():
    layer = DecoderLayer(d_model=128, num_heads=4, d_ff=512, dropout=0.0)
    x      = torch.randn(2, 7, 128)
    memory = torch.randn(2, 9, 128)
    src_mask = torch.zeros(2, 1, 1, 9, dtype=torch.bool)
    tgt_mask = torch.zeros(2, 1, 7, 7, dtype=torch.bool)
    out = layer(x, memory, src_mask, tgt_mask)
    assert out.shape == x.shape, f"DecoderLayer shape {out.shape}"
    return "DecoderLayer preserves [B, L_tgt, d_model] shape"


def test_encoder_stack_shape_and_independence():
    proto = EncoderLayer(d_model=128, num_heads=4, d_ff=512, dropout=0.0)
    enc = Encoder(proto, N=4)
    x = torch.randn(2, 9, 128)
    mask = torch.zeros(2, 1, 1, 9, dtype=torch.bool)
    out = enc(x, mask)
    assert out.shape == x.shape, f"Encoder shape {out.shape}"

    # Verify layers have independent weights (no shared-state from copy.deepcopy avoidance).
    w0 = enc.layers[0].self_attn.qkv.weight
    w1 = enc.layers[1].self_attn.qkv.weight
    assert w0.data_ptr() != w1.data_ptr(), "encoder layers share weight storage"
    assert not torch.equal(w0, w1), "encoder layers have identical weights"
    return "Encoder stack shape correct, layers are independent"


def test_decoder_stack_shape():
    proto = DecoderLayer(d_model=128, num_heads=4, d_ff=512, dropout=0.0)
    dec = Decoder(proto, N=4)
    x      = torch.randn(2, 7, 128)
    memory = torch.randn(2, 9, 128)
    src_mask = torch.zeros(2, 1, 1, 9, dtype=torch.bool)
    tgt_mask = torch.zeros(2, 1, 7, 7, dtype=torch.bool)
    out = dec(x, memory, src_mask, tgt_mask)
    assert out.shape == x.shape, f"Decoder shape {out.shape}"
    return "Decoder stack preserves shape"


def test_real_masks_propagate():
    # Use real make_src_mask / make_tgt_mask to ensure shape contracts line up.
    proto_enc = EncoderLayer(d_model=64, num_heads=4, d_ff=256, dropout=0.0)
    proto_dec = DecoderLayer(d_model=64, num_heads=4, d_ff=256, dropout=0.0)
    enc = Encoder(proto_enc, N=2)
    dec = Decoder(proto_dec, N=2)

    src = torch.tensor([[5, 6, 7, 1, 1], [3, 4, 1, 1, 1]])  # pad=1
    tgt = torch.tensor([[2, 5, 6, 1], [2, 3, 1, 1]])
    src_mask = make_src_mask(src, pad_idx=1)
    tgt_mask = make_tgt_mask(tgt, pad_idx=1)

    src_emb = torch.randn(2, 5, 64)
    tgt_emb = torch.randn(2, 4, 64)

    memory = enc(src_emb, src_mask)
    out    = dec(tgt_emb, memory, src_mask, tgt_mask)
    assert memory.shape == (2, 5, 64)
    assert out.shape    == (2, 4, 64)
    return "real masks flow through Encoder + Decoder correctly"


if __name__ == "__main__":
    tests = [
        test_ffn_shape,
        test_encoder_layer_shape,
        test_decoder_layer_shape,
        test_encoder_stack_shape_and_independence,
        test_decoder_stack_shape,
        test_real_masks_propagate,
    ]
    print(f"Running {len(tests)} layer smoke tests\n" + "-" * 60)
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
