"""
End-to-end test for Multi30kDataset + Vocab + collate_batch.
First run downloads Multi30k from HuggingFace (~5 MB).
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, ".")
from dataset import Multi30kDataset, Vocab, collate_batch


def test_vocab_specials_at_fixed_indices():
    v = Vocab.build_from_iterator([["hello", "world"], ["world", "again"]], min_freq=1)
    assert v["<unk>"] == 0 and v.itos[0] == "<unk>"
    assert v["<pad>"] == 1 and v.itos[1] == "<pad>"
    assert v["<sos>"] == 2 and v.itos[2] == "<sos>"
    assert v["<eos>"] == 3 and v.itos[3] == "<eos>"
    return "specials at fixed indices 0..3"


def test_vocab_encode_decode_roundtrip():
    v = Vocab.build_from_iterator([["foo", "bar", "baz"]], min_freq=1)
    ids = v.encode(["foo", "bar"], add_specials=True)
    assert ids[0] == Vocab.SOS_IDX and ids[-1] == Vocab.EOS_IDX
    tokens = v.decode(ids, strip_specials=True)
    assert tokens == ["foo", "bar"], f"roundtrip got {tokens}"
    return "encode/decode roundtrip preserves tokens"


def test_vocab_unk_for_oov():
    v = Vocab.build_from_iterator([["a", "b", "c"]], min_freq=1)
    assert v["nonexistent"] == Vocab.UNK_IDX
    return "OOV tokens map to <unk>"


def test_vocab_save_load_roundtrip(tmp_path=Path("artifacts/_test_vocab.pt")):
    v = Vocab.build_from_iterator([["one", "two", "three"]], min_freq=1)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    v.save(tmp_path)
    v2 = Vocab.load(tmp_path)
    assert v.itos == v2.itos and v.stoi == v2.stoi
    tmp_path.unlink()
    return "Vocab save/load roundtrip"


def test_multi30k_train_loads_and_builds_vocab():
    ds = Multi30kDataset(split="train", artifacts_dir="artifacts", max_len=128)
    assert len(ds) == 29000, f"expected 29000 train pairs, got {len(ds)}"

    src_vocab, tgt_vocab = ds.build_vocab(min_freq=2)
    assert 5000 < len(src_vocab) < 15000, f"de vocab size {len(src_vocab)} suspicious"
    assert 5000 < len(tgt_vocab) < 15000, f"en vocab size {len(tgt_vocab)} suspicious"

    # Saved artifacts exist
    assert (Path("artifacts") / "vocab_de.pt").exists()
    assert (Path("artifacts") / "vocab_en.pt").exists()
    return f"train: 29000 pairs, vocab sizes de={len(src_vocab)}, en={len(tgt_vocab)}"


def test_multi30k_getitem_shape_and_specials():
    ds = Multi30kDataset(split="train", artifacts_dir="artifacts", max_len=128)
    src, tgt = ds[0]
    assert src.dtype == torch.long and tgt.dtype == torch.long
    assert src[0].item() == Vocab.SOS_IDX, f"src must start with <sos>"
    assert src[-1].item() == Vocab.EOS_IDX, f"src must end with <eos>"
    assert tgt[0].item() == Vocab.SOS_IDX
    assert tgt[-1].item() == Vocab.EOS_IDX
    return f"__getitem__ returns wrapped ids (src_len={src.size(0)}, tgt_len={tgt.size(0)})"


def test_validation_and_test_splits():
    val_ds = Multi30kDataset(split="validation", artifacts_dir="artifacts", max_len=128)
    test_ds = Multi30kDataset(split="test", artifacts_dir="artifacts", max_len=128)
    assert len(val_ds)  == 1014, f"val size {len(val_ds)}"
    assert len(test_ds) == 1000, f"test size {len(test_ds)}"
    # Vocabs auto-loaded
    assert val_ds.src_vocab is not None and test_ds.tgt_vocab is not None
    return f"val={len(val_ds)}, test={len(test_ds)} with vocabs auto-loaded"


def test_collate_padding():
    ds = Multi30kDataset(split="train", artifacts_dir="artifacts", max_len=128)
    batch = [ds[i] for i in range(5)]
    src, tgt = collate_batch(batch, pad_idx=Vocab.PAD_IDX)

    assert src.shape[0] == 5 and tgt.shape[0] == 5
    # Every row must end with pad OR eos (no random garbage in pad region)
    for i in range(5):
        # The original src length
        orig_src_len = batch[i][0].size(0)
        assert torch.all(src[i, orig_src_len:] == Vocab.PAD_IDX), \
            f"row {i} pad region contains non-pad"
    return f"collate produces padded [{src.shape}, {tgt.shape}]"


def test_no_data_leakage_train_vs_test():
    """Sanity: train and test should NOT have overlapping sentences."""
    train = Multi30kDataset(split="train", artifacts_dir="artifacts")
    test  = Multi30kDataset(split="test",  artifacts_dir="artifacts")

    train_first_20 = {train._hf_split[i]["en"] for i in range(20)}
    test_first_20  = {test._hf_split[i]["en"]  for i in range(20)}
    overlap = train_first_20 & test_first_20
    assert not overlap, f"train/test leak detected: {overlap}"
    return "train and test splits show no leakage in first 20 samples"


if __name__ == "__main__":
    tests = [
        test_vocab_specials_at_fixed_indices,
        test_vocab_encode_decode_roundtrip,
        test_vocab_unk_for_oov,
        test_vocab_save_load_roundtrip,
        test_multi30k_train_loads_and_builds_vocab,
        test_multi30k_getitem_shape_and_specials,
        test_validation_and_test_splits,
        test_collate_padding,
        test_no_data_leakage_train_vs_test,
    ]
    print(f"Running {len(tests)} dataset tests\n" + "-" * 70)
    failed = 0
    for t in tests:
        try:
            print(f"  PASS  {t():<68}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}");  failed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}");  failed += 1
    print("-" * 70)
    msg = "all passed" if failed == 0 else f"{failed} failed"
    print(f"  {len(tests) - failed}/{len(tests)}  {msg}")
    sys.exit(0 if failed == 0 else 1)
