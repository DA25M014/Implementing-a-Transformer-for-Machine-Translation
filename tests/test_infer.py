"""
End-to-end test of greedy_decode and Transformer.infer.

The model has random weights here, so we don\\'t check translation
quality -- we verify that the pipeline runs end-to-end without
errors and produces a well-formed English string.
"""
import sys
import torch

sys.path.insert(0, ".")
from model import Transformer, make_src_mask
from train import greedy_decode
from dataset import Vocab


def test_greedy_decode_returns_growing_sequence():
    torch.manual_seed(25014)
    model = Transformer().eval()
    src = torch.tensor([[Vocab.SOS_IDX, 10, 20, 30, Vocab.EOS_IDX]])
    src_mask = make_src_mask(src, pad_idx=Vocab.PAD_IDX)

    ys = greedy_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_len=20,
        start_symbol=Vocab.SOS_IDX,
        end_symbol=Vocab.EOS_IDX,
        device="cpu",
    )
    assert ys.shape[0] == 1, f"batch dim should be 1, got {ys.shape}"
    assert ys[0, 0].item() == Vocab.SOS_IDX, "must start with <sos>"
    assert ys.shape[1] >= 2, "must contain at least <sos> and one token"
    assert ys.shape[1] <= 20, f"must not exceed max_len, got {ys.shape[1]}"
    return f"greedy_decode produced [1, {ys.shape[1]}] sequence"


def test_greedy_decode_stops_on_eos():
    """If <eos> is produced, decoding should stop there."""
    torch.manual_seed(0)
    model = Transformer().eval()
    src = torch.tensor([[Vocab.SOS_IDX, 5, Vocab.EOS_IDX]])
    src_mask = make_src_mask(src, pad_idx=Vocab.PAD_IDX)

    # Try several random seeds; for at least one we expect an early EOS
    # given random weights. We just check that IF the sequence ends with EOS,
    # nothing comes after it.
    ys = greedy_decode(
        model=model, src=src, src_mask=src_mask,
        max_len=30,
        start_symbol=Vocab.SOS_IDX,
        end_symbol=Vocab.EOS_IDX,
        device="cpu",
    )
    seq = ys[0].tolist()
    if Vocab.EOS_IDX in seq[1:]:
        eos_pos = seq.index(Vocab.EOS_IDX, 1)
        assert eos_pos == len(seq) - 1, "tokens generated after <eos>"
        return "greedy_decode stops on <eos>"
    return "no <eos> emitted; sequence ran to max_len (acceptable)"


def test_infer_end_to_end():
    """Autograder contract: model = Transformer(); model.eval(); model.infer(text)"""
    torch.manual_seed(25014)
    model = Transformer().to("cpu")
    model.eval()
    out = model.infer("Ein Mann läuft im Park.")
    assert isinstance(out, str), f"infer should return str, got {type(out)}"
    assert "<sos>" not in out and "<eos>" not in out and "<pad>" not in out
    return f"infer end-to-end returns string: {out!r}"


def test_infer_handles_short_input():
    torch.manual_seed(25014)
    model = Transformer().eval()
    out = model.infer("Hallo.")
    assert isinstance(out, str)
    return f"short input handled: {out!r}"


def test_infer_handles_long_input():
    """Input longer than max_len should be truncated, not error."""
    torch.manual_seed(25014)
    model = Transformer().eval()
    long_sentence = "Ein Mann " * 100
    out = model.infer(long_sentence)
    assert isinstance(out, str)
    return f"long input handled (output len={len(out.split())} tokens)"


def test_infer_does_not_modify_model_state():
    """A side-effect-free contract: infer should not flip training mode etc."""
    torch.manual_seed(25014)
    model = Transformer().eval()
    is_training_before = model.training
    _ = model.infer("Ein Test.")
    assert model.training == is_training_before, "infer flipped training mode"
    return "infer is side-effect-free w.r.t. training mode"


if __name__ == "__main__":
    tests = [
        test_greedy_decode_returns_growing_sequence,
        test_greedy_decode_stops_on_eos,
        test_infer_end_to_end,
        test_infer_handles_short_input,
        test_infer_handles_long_input,
        test_infer_does_not_modify_model_state,
    ]
    print(f"Running {len(tests)} infer/greedy_decode tests\n" + "-" * 78)
    failed = 0
    for t in tests:
        try:
            print(f"  PASS  {t():<76}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}");  failed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}");  failed += 1
    print("-" * 78)
    msg = "all passed" if failed == 0 else f"{failed} failed"
    print(f"  {len(tests) - failed}/{len(tests)}  {msg}")
    sys.exit(0 if failed == 0 else 1)
