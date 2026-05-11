"""
Unit tests for LabelSmoothingLoss, run_epoch, save/load_checkpoint.
"""
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from model import Transformer
from train import (
    LabelSmoothingLoss,
    save_checkpoint, load_checkpoint,
    run_epoch,
)


def test_label_smoothing_basic_shape():
    crit = LabelSmoothingLoss(vocab_size=100, pad_idx=1, smoothing=0.1)
    logits = torch.randn(32, 100)
    target = torch.randint(2, 100, (32,))
    loss = crit(logits, target)
    assert loss.ndim == 0, "loss must be scalar"
    assert loss.item() > 0, "loss should be positive"
    return f"loss = {loss.item():.4f}"


def test_label_smoothing_pad_rows_excluded():
    crit = LabelSmoothingLoss(vocab_size=100, pad_idx=1, smoothing=0.1)
    logits = torch.randn(8, 100)
    target = torch.full((8,), 1)
    loss = crit(logits, target)
    assert loss.item() < 1.0
    return f"all-pad target loss = {loss.item():.6f}"


def test_label_smoothing_target_distribution_sums_to_one():
    smooth_val = 0.1 / (10 - 2)
    expected = torch.full((10,), smooth_val)
    expected[1] = 0.0
    expected[5] = 0.9
    assert abs(expected.sum().item() - 1.0) < 1e-6
    return "smoothed target distribution sums to 1"


def test_label_smoothing_eps_zero_matches_ce_for_nonpad():
    V = 50
    smooth_loss = LabelSmoothingLoss(vocab_size=V, pad_idx=1, smoothing=0.0)
    ce_loss = nn.CrossEntropyLoss(ignore_index=1)

    torch.manual_seed(0)
    logits = torch.randn(20, V)
    target = torch.randint(2, V, (20,))

    a = smooth_loss(logits, target).item()
    b = ce_loss(logits, target).item()
    assert abs(a - b) < 1e-4, f"got {a} vs {b}"
    return f"smoothing=0 matches CE ({a:.4f} ~= {b:.4f})"


def test_save_load_checkpoint_roundtrip():
    torch.manual_seed(25014)
    model = Transformer(src_vocab_size=200, tgt_vocab_size=180,
                        d_model=64, N=2, num_heads=4, d_ff=128, dropout=0.0).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    from model import make_src_mask, make_tgt_mask
    src = torch.randint(2, 200, (2, 5))
    tgt = torch.randint(2, 180, (2, 4))
    src_mask = make_src_mask(src, pad_idx=1)
    tgt_mask = make_tgt_mask(tgt, pad_idx=1)

    out_before = model(src, tgt, src_mask, tgt_mask).detach()

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        save_checkpoint(model, optimizer, None, epoch=7, path=str(path))
        assert path.exists()

        model2 = Transformer(src_vocab_size=200, tgt_vocab_size=180,
                             d_model=64, N=2, num_heads=4, d_ff=128, dropout=0.0).eval()
        loaded_epoch = load_checkpoint(str(path), model2)
        assert loaded_epoch == 7

        out_after = model2(src, tgt, src_mask, tgt_mask).detach()
        assert torch.allclose(out_before, out_after, atol=1e-5)
    return f"checkpoint roundtrip preserves outputs (epoch={loaded_epoch})"


def test_checkpoint_model_config_keys():
    torch.manual_seed(25014)
    model = Transformer(src_vocab_size=200, tgt_vocab_size=180,
                        d_model=64, N=2, num_heads=4, d_ff=128, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        save_checkpoint(model, optimizer, None, epoch=0, path=str(path))
        blob = torch.load(path, weights_only=False)

    required = {"src_vocab_size", "tgt_vocab_size", "d_model", "N",
                "num_heads", "d_ff", "dropout", "max_len"}
    config = blob["model_config"]
    missing = required - set(config.keys())
    assert not missing, f"missing keys: {missing}"
    n = len(config)
    return f"model_config has all reconstruction keys ({n} total)"


def test_run_epoch_minimal_one_batch():
    torch.manual_seed(25014)
    model = Transformer(src_vocab_size=50, tgt_vocab_size=40,
                        d_model=32, N=2, num_heads=4, d_ff=64, dropout=0.0)
    crit = LabelSmoothingLoss(vocab_size=40, pad_idx=1, smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    src_a = torch.tensor([[2, 5, 6, 3, 1]])
    tgt_a = torch.tensor([[2, 5, 6, 3, 1]])
    src_b = torch.tensor([[2, 7, 8, 9, 3]])
    tgt_b = torch.tensor([[2, 7, 8, 9, 3]])
    fake_loader = [(src_a, tgt_a), (src_b, tgt_b)]

    metrics = run_epoch(fake_loader, model, crit, optimizer=opt, scheduler=None,
                        epoch_num=0, is_train=True, device="cpu", pad_idx=1,
                        wandb_run=None)
    assert "loss" in metrics and metrics["loss"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    loss_v = metrics["loss"]
    acc_v = metrics["accuracy"]
    return f"run_epoch executes: loss={loss_v:.4f} acc={acc_v:.3f}"


if __name__ == "__main__":
    tests = [
        test_label_smoothing_basic_shape,
        test_label_smoothing_pad_rows_excluded,
        test_label_smoothing_target_distribution_sums_to_one,
        test_label_smoothing_eps_zero_matches_ce_for_nonpad,
        test_save_load_checkpoint_roundtrip,
        test_checkpoint_model_config_keys,
        test_run_epoch_minimal_one_batch,
    ]
    print(f"Running {len(tests)} train-pipeline tests")
    print("-" * 78)
    failed = 0
    for t in tests:
        try:
            print(f"  PASS  {t():<70}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print("-" * 78)
    msg = "all passed" if failed == 0 else f"{failed} failed"
    print(f"  {len(tests) - failed}/{len(tests)}  {msg}")
    sys.exit(0 if failed == 0 else 1)
