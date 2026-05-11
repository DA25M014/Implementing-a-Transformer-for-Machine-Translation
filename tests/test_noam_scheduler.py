"""
Unit tests for NoamScheduler.
Run from repo root:  python tests/test_noam_scheduler.py
Mirrors the 5 autograder criteria for the [10M] Noam grading block.
"""
import math
import sys
import torch
import torch.optim as optim

sys.path.insert(0, ".")
from lr_scheduler import NoamScheduler, get_lr_history


D_MODEL      = 512
WARMUP_STEPS = 4000
TOTAL_STEPS  = 12000


def _history(d_model=D_MODEL, warmup=WARMUP_STEPS, total=TOTAL_STEPS):
    return get_lr_history(d_model, warmup, total)


def test_monotonic_increase_during_warmup():
    lrs = _history()
    warmup_lrs = lrs[:WARMUP_STEPS]
    diffs = [warmup_lrs[i+1] - warmup_lrs[i] for i in range(len(warmup_lrs) - 1)]
    assert all(d > 0 for d in diffs), \
        f"LR not strictly increasing during warmup; min diff {min(diffs)}"
    return "LR monotonically increases during warmup"


def test_peak_within_10_steps_of_warmup():
    lrs = _history()
    peak_step = max(range(len(lrs)), key=lambda i: lrs[i])
    delta = abs(peak_step - WARMUP_STEPS)
    assert delta <= 10, \
        f"peak at step {peak_step}, expected within 10 of {WARMUP_STEPS}"
    return f"peak at step {peak_step} (within 10 of warmup={WARMUP_STEPS})"


def test_monotonic_decrease_after_warmup():
    lrs = _history()
    # Start a few steps past warmup to skip the exact crossover point
    post = lrs[WARMUP_STEPS + 5:]
    diffs = [post[i+1] - post[i] for i in range(len(post) - 1)]
    assert all(d < 0 for d in diffs), \
        f"LR not strictly decreasing post-warmup; max diff {max(diffs)}"
    return "LR monotonically decreases after warmup"


def test_peak_matches_closed_form():
    lrs = _history()
    peak = max(lrs)
    expected = (D_MODEL ** -0.5) * (WARMUP_STEPS ** -0.5)
    rel_err = abs(peak - expected) / expected
    assert rel_err < 1e-4, \
        f"peak {peak:.6e} vs closed-form {expected:.6e} (rel err {rel_err:.2e})"
    return f"peak {peak:.6e} matches closed-form (rel err {rel_err:.2e})"


def test_lr_at_step_one_matches_formula():
    # Build scheduler manually so we can inspect step 1 explicitly
    dummy = torch.nn.Linear(1, 1)
    opt   = optim.Adam(dummy.parameters(), lr=1.0)
    sched = NoamScheduler(opt, d_model=D_MODEL, warmup_steps=WARMUP_STEPS)

    # After construction, last_epoch is 0 (LRScheduler.__init__ calls step once).
    # The LR currently in the param group is for step 1.
    lr_step1 = opt.param_groups[0]["lr"]

    step = 1
    expected = (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))
    rel_err = abs(lr_step1 - expected) / expected
    assert rel_err < 1e-6, \
        f"LR at step 1 = {lr_step1:.6e}, expected {expected:.6e}"
    return f"LR at step 1 = {lr_step1:.6e} matches formula"


if __name__ == "__main__":
    tests = [
        test_monotonic_increase_during_warmup,
        test_peak_within_10_steps_of_warmup,
        test_monotonic_decrease_after_warmup,
        test_peak_matches_closed_form,
        test_lr_at_step_one_matches_formula,
    ]
    print(f"Running {len(tests)} Noam tests\n" + "-" * 60)
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
