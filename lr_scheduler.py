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
    Noam learning rate schedule from "Attention Is All You Need".

    Two arms:
        arm_warmup = step * warmup_steps^(-1.5)   (linear warmup)
        arm_decay  = step^(-0.5)                  (inverse-sqrt decay)

    The effective scale at step t is:
        scale(t) = d_model^(-0.5) * min(arm_decay, arm_warmup)

    The crossover happens exactly at step == warmup_steps, where both
    arms equal warmup_steps^(-0.5). The peak LR is therefore:
        lr_peak = d_model^(-0.5) * warmup_steps^(-0.5)

    Because this scheduler multiplies each param group's base_lr by
    this scale, initialize your optimizer with lr=1.0 so the schedule
    produces raw Noam values (not a scaled-down version).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")

        self.d_model      = d_model
        self.warmup_steps = warmup_steps

        # Precompute the model-dim factor; reused on every step.
        self._dim_factor = d_model ** -0.5

        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        step = self.last_epoch + 1                          # avoid step == 0

        arm_decay  = step ** -0.5
        arm_warmup = step * (self.warmup_steps ** -1.5)

        return self._dim_factor * min(arm_decay, arm_warmup)

    def get_lr(self) -> list[float]:
        if not self.base_lrs:
            return []
        scale = self._get_lr_scale()
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
