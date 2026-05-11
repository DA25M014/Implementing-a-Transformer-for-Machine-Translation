"""
Verify the autograder contract for Transformer.__init__:
  - Bare Transformer() instantiates
  - Vocab + tokenizers auto-load from artifacts/
  - Weights NOT downloaded when CHECKPOINT_GDRIVE_ID is None (default)
"""
import sys
import torch

sys.path.insert(0, ".")
from model import Transformer


def test_bare_instantiation_loads_vocab_and_tokenizers():
    model = Transformer()
    assert model.src_vocab is not None, "src_vocab not loaded"
    assert model.tgt_vocab is not None, "tgt_vocab not loaded"
    assert model.src_tokenizer is not None, "src_tokenizer not loaded"
    assert model.tgt_tokenizer is not None, "tgt_tokenizer not loaded"

    # Tokenizers should actually work.
    de_tokens = model.src_tokenizer("Ein Mann läuft im Park.")
    en_tokens = model.tgt_tokenizer("A man runs in the park.")
    assert len(de_tokens) > 0 and len(en_tokens) > 0
    return f"vocab + tokenizers loaded (de_vocab={len(model.src_vocab)}, en_vocab={len(model.tgt_vocab)})"


def test_defaults_match_real_vocab_sizes():
    model = Transformer()
    # The class defaults should be in-range with the actual vocab files.
    assert abs(model.src_vocab_size - len(model.src_vocab)) <= 1, \
        f"src_vocab_size default {model.src_vocab_size} vs actual {len(model.src_vocab)}"
    assert abs(model.tgt_vocab_size - len(model.tgt_vocab)) <= 1, \
        f"tgt_vocab_size default {model.tgt_vocab_size} vs actual {len(model.tgt_vocab)}"
    return f"defaults aligned with on-disk vocabs"


def test_eval_mode_and_to_device():
    """The autograder runs:  model = Transformer().to(device); model.eval()"""
    model = Transformer().to("cpu")
    model.eval()
    assert not model.training
    return ".to(device) and .eval() work as the autograder uses them"


def test_no_gdown_call_without_id():
    """CHECKPOINT_GDRIVE_ID is None by default; no network call should happen."""
    # If gdown were called with id=None, it would raise. Successful instantiation
    # is sufficient evidence that the download path was correctly skipped.
    Transformer()
    return "no gdown invocation when CHECKPOINT_GDRIVE_ID is None"


def test_infer_returns_string():
    """After Step 20, infer() is wired end-to-end; should return a string."""
    model = Transformer().eval()
    out = model.infer("Ein Test.")
    assert isinstance(out, str), f"expected str, got {type(out).__name__}"
    return f"infer() returns string: {out!r}"


def test_param_count_with_real_defaults():
    model = Transformer()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Transformer-base on Multi30k with tied embeddings: roughly 40-60M params.
    assert 30_000_000 < n < 80_000_000, f"param count {n:,} outside expected range"
    return f"param count {n:,} reasonable for d_model=512, N=6"


if __name__ == "__main__":
    tests = [
        test_bare_instantiation_loads_vocab_and_tokenizers,
        test_defaults_match_real_vocab_sizes,
        test_eval_mode_and_to_device,
        test_no_gdown_call_without_id,
        test_infer_returns_string,
        test_param_count_with_real_defaults,
    ]
    print(f"Running {len(tests)} Transformer __init__ tests\n" + "-" * 70)
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
