"""
train.py -- Training pipeline, evaluation, checkpointing.

Implements the autograder contract:
    greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device)
    evaluate_bleu(model, test_dataloader, tgt_vocab, device)
    save_checkpoint(model, optimizer, scheduler, epoch, path)
    load_checkpoint(path, model, optimizer, scheduler) -> int

Plus the orchestration:
    LabelSmoothingLoss   -- KL-divergence form, eps/(V-1) mass on wrong classes.
    run_epoch            -- one train or eval epoch with W&B logging hooks.
    run_training_experiment(config) -- single entrypoint for all 5 ablation runs.
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing via explicit smoothed-target KL divergence.

    Smoothed distribution per token:
        true_class       -> (1 - smoothing)
        any other class  -> smoothing / (V - 2)        (V - 2 because pad also excluded)
        pad class        -> 0

    Note: the skeleton docstring states eps/(V-1), but pad is also
    excluded from the smoothed mass, so the practical denominator is
    V-2. The difference is negligible for V > 1000.

    Loss is the KL divergence sum over the vocabulary axis, averaged
    over non-pad target tokens.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 1, smoothing: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.criterion  = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [N, V] raw logits (typically logits.reshape(-1, V))
            target : [N]    gold ids

        Returns:
            scalar loss, averaged over non-pad tokens
        """
        assert logits.size(1) == self.vocab_size,             f"logits dim {logits.size(1)} != vocab {self.vocab_size}"

        # Build the smoothed target distribution.
        with torch.no_grad():
            smooth_val = self.smoothing / (self.vocab_size - 2)
            true_dist = torch.full_like(logits, smooth_val)
            true_dist[:, self.pad_idx] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            # Rows where the target IS pad get all-zero distribution (masked out).
            pad_rows = (target == self.pad_idx)
            true_dist[pad_rows] = 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        loss_sum  = self.criterion(log_probs, true_dist)

        n_nonpad = (~pad_rows).sum().clamp(min=1)
        return loss_sum / n_nonpad


# ══════════════════════════════════════════════════════════════════════
#  TRAINING / EVAL EPOCH
# ══════════════════════════════════════════════════════════════════════

def _qk_grad_norms(model: Transformer) -> tuple[float, float]:
    """
    Aggregate the Frobenius norms of all Q and K projection gradients
    in the encoder + decoder self-attention layers.

    Returns (q_norm, k_norm). Each is the L2 norm of the concatenated
    grads from every layer. Used for the section 2.2 ablation analysis.
    """
    q_grads, k_grads = [], []
    for module_path, module in model.named_modules():
        if not module_path.endswith("self_attn"):
            continue
        if module.qkv.weight.grad is None:
            continue
        d = module.d_model
        full_grad = module.qkv.weight.grad
        q_grads.append(full_grad[:d].flatten())
        k_grads.append(full_grad[d:2*d].flatten())

    if not q_grads:
        return 0.0, 0.0

    q_norm = torch.cat(q_grads).norm().item()
    k_norm = torch.cat(k_grads).norm().item()
    return q_norm, k_norm


def _prediction_confidence(logits: torch.Tensor, target: torch.Tensor, pad_idx: int = 1) -> float:
    """
    Mean softmax probability assigned to the correct token, averaged over
    non-pad positions. Used for the section 2.5 label-smoothing analysis.
    """
    probs = F.softmax(logits, dim=-1)
    correct_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = (target != pad_idx)
    if mask.sum().item() == 0:
        return 0.0
    return (correct_probs * mask).sum().item() / mask.sum().item()


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    pad_idx: int = 1,
    log_every: int = 50,
    wandb_run=None,
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> dict[str, float]:
    """
    Run one epoch of training or evaluation.

    Returns a dict with: 'loss', 'perplexity', 'accuracy'.
    On train epochs, also logs per-step metrics to W&B if wandb_run is provided.
    """
    model.train(is_train)

    total_loss, total_tokens, total_correct = 0.0, 0, 0
    step_in_epoch = 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        # Decoder input is the target shifted right; the loss target is shifted left.
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:,  1:]

        src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx=pad_idx).to(device)

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (use_amp and device == "cuda")
            else torch.amp.autocast(device_type="cpu", enabled=False)
        )

        with amp_ctx:
            logits = model(src, tgt_in, src_mask, tgt_mask)
            V      = logits.size(-1)
            loss   = loss_fn(logits.reshape(-1, V), tgt_out.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()

            q_norm, k_norm = _qk_grad_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if wandb_run is not None and step_in_epoch % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                conf = _prediction_confidence(logits.detach(), tgt_out, pad_idx=pad_idx)
                wandb_run.log({
                    "train/loss":            loss.item(),
                    "train/step_lr":         lr,
                    "train/grad_norm_Q":     q_norm,
                    "train/grad_norm_K":     k_norm,
                    "train/pred_confidence": conf,
                    "epoch":                 epoch_num,
                })

        # Accumulate metrics over non-pad tokens.
        with torch.no_grad():
            mask = (tgt_out != pad_idx)
            n_tok = mask.sum().item()
            preds = logits.argmax(dim=-1)
            n_correct = ((preds == tgt_out) & mask).sum().item()

            total_loss    += loss.item() * n_tok
            total_tokens  += n_tok
            total_correct += n_correct

        step_in_epoch += 1

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss":       avg_loss,
        "perplexity": math.exp(min(avg_loss, 20)),     # cap to avoid overflow
        "accuracy":   total_correct / max(total_tokens, 1),
    }


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING  (kept from Step 20)
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Token-by-token greedy decoding from a trained Transformer.
    Encoder runs once outside the loop; decoder is called per step
    on a growing prefix.
    """
    model = model.to(device)
    src      = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)

    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
        logits   = model.decode(memory, src_mask, ys, tgt_mask)
        next_logits = logits[:, -1, :]
        next_id     = next_logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_id], dim=1)
        if next_id.item() == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
    raw_references: list = None,
) -> float:
    """
    Corpus-level BLEU via sacrebleu, computed by greedy-decoding each
    source sentence.

    IMPORTANT: When raw_references is supplied (list of original English
    strings from the HuggingFace dataset), BLEU is computed against
    THOSE raw strings using the model\'s detokenized output. This matches
    what the autograder does and gives an honest BLEU number.

    When raw_references is None, we fall back to vocab-roundtripped
    references (the old behaviour, kept for backwards compatibility
    in unit tests). That mode reports inflated BLEU because rare words
    become <unk> on both sides and get stripped.
    """
    import sacrebleu

    model = model.to(device).eval()
    hypotheses, references = [], []
    sentence_idx = 0

    with torch.no_grad():
        for src, tgt in test_dataloader:
            for i in range(src.size(0)):
                s = src[i:i+1].to(device)
                t = tgt[i].tolist()

                s_mask = make_src_mask(s, pad_idx=1).to(device)
                y = greedy_decode(
                    model=model, src=s, src_mask=s_mask,
                    max_len=max_len,
                    start_symbol=tgt_vocab.SOS_IDX,
                    end_symbol=tgt_vocab.EOS_IDX,
                    device=device,
                )
                hyp_tokens = tgt_vocab.decode(y[0].tolist(), strip_specials=True)
                hyp_str    = " ".join(hyp_tokens)
                # Apply the same detokenisation as Transformer.infer
                hyp_str    = Transformer._detokenize(hyp_str)
                hypotheses.append(hyp_str)

                if raw_references is not None:
                    references.append(raw_references[sentence_idx])
                else:
                    ref_tokens = tgt_vocab.decode(t, strip_specials=True)
                    references.append(" ".join(ref_tokens))
                sentence_idx += 1

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return float(bleu.score)


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state, plus the model_config dict
    needed to reconstruct the architecture at load time.
    """
    model_config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":        model.encoder.layers[0].drop.p,
        "max_len":        model.max_len,
    }
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "model_config":         model_config,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.
    Returns the saved epoch number.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    if optimizer is not None and blob.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(blob["optimizer_state_dict"])
    if scheduler is not None and blob.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(blob["scheduler_state_dict"])
    return int(blob.get("epoch", 0))


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Architecture
    "d_model":        512,
    "N":              6,
    "num_heads":      8,
    "d_ff":           2048,
    "dropout":        0.15,
    "max_len":        128,

    # Optimization
    "batch_size":     128,
    "num_epochs":     25,
    "warmup_steps":   4500,
    "eval_every":     1,           # epochs between val BLEU evals
    "use_amp":        False,       # bf16 autocast (CUDA only)
    "early_stop_patience": 5,      # stop if val BLEU does not improve for N evals
    "label_smooth":   0.1,
    "betas":          (0.9, 0.98),
    "eps":            1e-9,
    "grad_clip":      1.0,
    "seed":           25014,

    # Ablation switches
    "scheduler":      "noam",          # "noam" | "fixed"
    "fixed_lr":       1e-4,            # used when scheduler == "fixed"
    "use_scaling":    True,            # set False for sec 2.2 ablation
    "pos_encoding":   "sinusoidal",    # "sinusoidal" | "learned" -- sec 2.4

    # Run identity
    "run_name":       "main",
    "wandb_project":  "da6401-a3-transformer",
    "wandb_group":    "baseline",
    "wandb_tags":     ["control"],

    # I/O
    "artifacts_dir":  "artifacts",
    "checkpoint":     "transformer_main.pt",
}


def run_training_experiment(config: dict | None = None) -> dict:
    """
    Single entrypoint for all 5 ablation runs.

    The 'config' dict is merged on top of DEFAULT_CONFIG, so a run can be
    fully specified by overriding only the keys that differ from main.

    Returns the final dict from the best epoch (by val BLEU).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    torch.manual_seed(cfg["seed"])

    # ── W&B (optional) ──────────────────────────────────────────────
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project=cfg["wandb_project"],
            name=cfg["run_name"],
            group=cfg["wandb_group"],
            tags=cfg["wandb_tags"],
            config=cfg,
            job_type="train",
        )
    except Exception as e:
        print(f"[wandb] disabled: {e}")

    # ── Data ────────────────────────────────────────────────────────
    from dataset import Multi30kDataset, Vocab, collate_batch

    train_ds = Multi30kDataset(split="train",      artifacts_dir=cfg["artifacts_dir"],
                               max_len=cfg["max_len"])
    if train_ds.src_vocab is None:
        train_ds.build_vocab(min_freq=2)
    val_ds  = Multi30kDataset(split="validation", artifacts_dir=cfg["artifacts_dir"],
                               max_len=cfg["max_len"])
    test_ds = Multi30kDataset(split="test",       artifacts_dir=cfg["artifacts_dir"],
                               max_len=cfg["max_len"])

    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_idx=1), num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              collate_fn=lambda b: collate_batch(b, pad_idx=1), num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              collate_fn=lambda b: collate_batch(b, pad_idx=1), num_workers=0)

    # ── Model ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg["d_model"], N=cfg["N"], num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"], dropout=cfg["dropout"], max_len=cfg["max_len"],
    ).to(device)

    # Optional ablation: replace sinusoidal PE with learned embedding.
    if cfg["pos_encoding"] == "learned":
        model.pos_enc = _build_learned_pe(cfg["d_model"], cfg["max_len"], cfg["dropout"]).to(device)

    # Optional ablation: disable 1/sqrt(d_k) scaling in attention.
    if not cfg["use_scaling"]:
        _disable_attention_scaling(model)

    # ── Optimization ────────────────────────────────────────────────
    base_lr = 1.0 if cfg["scheduler"] == "noam" else cfg["fixed_lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr,
                                 betas=cfg["betas"], eps=cfg["eps"])

    scheduler = None
    if cfg["scheduler"] == "noam":
        from lr_scheduler import NoamScheduler
        scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"],
                                  warmup_steps=cfg["warmup_steps"])

    loss_fn = LabelSmoothingLoss(vocab_size=len(tgt_vocab), pad_idx=1,
                                 smoothing=cfg["label_smooth"])

    # Pull raw English references from HF dataset for honest BLEU.
    raw_val_refs  = [val_ds._hf_split[i]["en"]  for i in range(len(val_ds))]
    raw_test_refs = [test_ds._hf_split[i]["en"] for i in range(len(test_ds))]

    # ── Training loop ───────────────────────────────────────────────
    best_bleu = -1.0
    best_epoch = -1
    no_improve_evals = 0
    history = []

    for epoch in range(cfg["num_epochs"]):
        train_metrics = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                                  epoch_num=epoch, is_train=True, device=device,
                                  pad_idx=1, wandb_run=wandb_run, grad_clip=cfg["grad_clip"],
                                  use_amp=cfg["use_amp"])
        val_metrics   = run_epoch(val_loader, model, loss_fn, optimizer=None, scheduler=None,
                                  epoch_num=epoch, is_train=False, device=device,
                                  pad_idx=1, wandb_run=None, use_amp=cfg["use_amp"])

        # Val BLEU only every cfg["eval_every"] epochs (and always on the last)
        do_eval = ((epoch % cfg["eval_every"]) == 0) or (epoch == cfg["num_epochs"] - 1)
        if do_eval:
            val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device,
                                     max_len=cfg["max_len"], raw_references=raw_val_refs)
        else:
            val_bleu = float("nan")
        history.append({"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}, "val_bleu": val_bleu})

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_bleu={val_bleu:.2f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch":            epoch,
                "val/loss":         val_metrics["loss"],
                "val/perplexity":   val_metrics["perplexity"],
                "val/accuracy":     val_metrics["accuracy"],
                "val/bleu":         val_bleu,
            })

        if do_eval:
            if val_bleu > best_bleu:
                best_bleu  = val_bleu
                best_epoch = epoch
                no_improve_evals = 0
                save_checkpoint(model, optimizer, scheduler, epoch, path=cfg["checkpoint"])
            else:
                no_improve_evals += 1
                if no_improve_evals >= cfg["early_stop_patience"]:
                    print(f"Early stop at epoch {epoch} "
                          f"(no improvement for {no_improve_evals} evals)")
                    break

    # ── Final test BLEU ─────────────────────────────────────────────
    load_checkpoint(cfg["checkpoint"], model)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device,
                              max_len=cfg["max_len"], raw_references=raw_test_refs)
    print(f"[final] best_epoch={best_epoch} best_val_bleu={best_bleu:.2f} test_bleu={test_bleu:.2f}")

    if wandb_run is not None:
        wandb_run.log({"test_bleu": test_bleu, "best_val_bleu": best_bleu, "best_epoch": best_epoch})
        wandb_run.finish()

    return {"best_epoch": best_epoch, "best_val_bleu": best_bleu, "test_bleu": test_bleu, "history": history}


# ── Ablation helpers ────────────────────────────────────────────────

class _LearnedPositionalEncoding(nn.Module):
    """For section 2.4 ablation: learned positional embeddings."""
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.drop  = nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        positions = torch.arange(L, device=x.device)
        return self.drop(x + self.embed(positions).unsqueeze(0))


def _build_learned_pe(d_model: int, max_len: int, dropout: float) -> nn.Module:
    return _LearnedPositionalEncoding(d_model, max_len, dropout)


def _disable_attention_scaling(model: Transformer) -> None:
    """
    Monkey-patch scaled_dot_product_attention to remove 1/sqrt(d_k) scaling.
    Used for section 2.2 ablation. Applied PROCESS-WIDE so it persists for
    the whole run; restore by restarting the process.
    """
    import model as model_mod

    def no_scale_attn(Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))   # no * scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        return torch.matmul(alpha, V), alpha

    model_mod.scaled_dot_product_attention = no_scale_attn


if __name__ == "__main__":
    run_training_experiment()
