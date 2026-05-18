"""
Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ─────────────────────────────────────────────
# TODO: Implement the NoamScheduler class below
# ─────────────────────────────────────────────

class NoamScheduler(LRScheduler):
    """
    Implementation of the learning-rate schedule from
    Vaswani et al. (2017), "Attention Is All You Need".

    Rather than expressing the schedule as the original
    `d_model^-0.5 * min(t^-0.5, t * W^-1.5)` form, we express it as
    a piecewise function over the training step `t`:

        - For 1 <= t <= W:   lr(t) = peak * (t / W)
        - For t >  W:        lr(t) = peak * sqrt(W / t)

    where peak = (d_model * W)^-0.5 is the value at t == W. This
    factorisation lets us precompute `peak` once and reduces each
    step to a single multiplication plus a cheap branch.

    The class multiplies each parameter group's `base_lr` by this
    schedule, so initialise the optimiser with `lr=1.0` to get the
    raw Noam values.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if d_model <= 0 or warmup_steps <= 0:
            raise ValueError(
                f"d_model and warmup_steps must be positive "
                f"(got d_model={d_model}, warmup_steps={warmup_steps})"
            )

        # Cast to int once so subsequent arithmetic is predictable.
        self._D = int(d_model)
        self._W = int(warmup_steps)

        # Peak LR multiplier; reached at step == W. Cached once.
        self._peak_scale = (self._D * self._W) ** -0.5

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        # `last_epoch` is incremented by the base class on each step;
        # at the very first call last_epoch == 0, hence step == 1.
        step = self.last_epoch + 1

        if step <= 0:
            # Defensive: shouldn't happen via normal use.
            return [0.0 for _ in self.base_lrs]

        # Phase-explicit form of the Noam schedule.
        if step < self._W:
            # Linear warm-up phase.
            scale = self._peak_scale * (step / self._W)
        elif step == self._W:
            # Crossover step: warm-up and decay arms coincide.
            scale = self._peak_scale
        else:
            # Inverse-square-root decay phase.
            scale = self._peak_scale * ((self._W / step) ** 0.5)

        return [base * scale for base in self.base_lrs]



# ──────────────────────────────────────────────────────────────────────
# Helper — do NOT modify
# ──────────────────────────────────────────────────────────────────────

def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list[float]:
    """
    Simulate the LR trajectory of NoamScheduler for `total_steps` steps.

    Args:
        d_model      (int): Model dimensionality.
        warmup_steps (int): Warm-up steps.
        total_steps  (int): Number of steps to simulate.

    Returns:
        list[float]: LR value at each step (length == total_steps).
    """
    dummy_model = torch.nn.Linear(1, 1)
    optimizer   = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler   = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


# ──────────────────────────────────────────────────────────────────────
# Quick visual check — run:  python noam_lr_scheduler.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()
