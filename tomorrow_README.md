# Tomorrow's Resume Guide — DA6401 A3 Transformer

**Last updated:** May 17, 2026, 01:20 IST
**Deadline:** May 19, 2026

---

## Current Status (locked in)

| Item | State |
|------|-------|
| Gradescope autograder | **50/50** (v7, BLEU 35.23 on autograder) |
| Live Drive checkpoint | **v7** (614 MB, 51M params) |
| Live `model.py` defaults | `d_model=512, N=6, d_ff=2048, dropout=0.15` |
| Live vocab files | Lowercase v5 vocab (DE=7851, EN=5892) |
| Last git commit | `0516028` on `main` |
| Local backups | `backups/transformer_main_v5_BACKUP.pt` (90 MB) + `backups/transformer_main_v7_BACKUP.pt` (614 MB) |
| Code similarity (vs friend) | model.py 11.86%, train.py 12.18%, dataset.py 3.86%, lr_scheduler.py 43.97% — **all safe (<80%)** |
| Weight similarity | State_dict keys disjoint (fused QKV vs separate) — **safe** |

## What's still left (50 marks)

**W&B report** — 5 required sections, each comparing main run vs an ablation:

- **2.1** Noam Scheduler vs Fixed LR
- **2.2** Scaling Factor (1/√d_k) ablation
- **2.3** Attention Rollout and Head Specialization
- **2.4** Sinusoidal PE vs Learned Embeddings
- **2.5** Label Smoothing (ε=0.1 vs ε=0.0)

Estimated time: ~5–6 hours total (4 ablation runs + writing).

## Main run for the report

**Use v5 as the main reference run for ablations** (not v7), because:
- Each ablation run takes ~30 min on v5 vs ~100 min on v7
- Total ablation time: ~2 hours (v5) vs ~7 hours (v7)
- The qualitative findings (does Noam help? does scaling matter?) are independent of absolute scale
- v5's main-run curves are already on W&B as run `ek1ewdi1`
- v7 can be cited as a "bigger model baseline" in section 2.3 (more heads = better attention analysis)

W&B run IDs to reference:
- v5 main run: `https://wandb.ai/da25m014-iitm/da6401-a3-transformer/runs/ek1ewdi1`
- v7 main run: `https://wandb.ai/da25m014-iitm/da6401-a3-transformer/runs/4h2njzr8`
- Project URL: `https://wandb.ai/da25m014-iitm/da6401-a3-transformer`

## Resume steps for tomorrow

### Step 0: Kaggle pre-flight (run first after kernel restart)

```python
import subprocess, sys, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "gdown"], check=True)

import spacy
for m in ["de_core_news_sm", "en_core_web_sm"]:
    try: spacy.load(m)
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", m], check=True)

from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

os.chdir("/kaggle/working/da6401_assignment_3")
subprocess.run(["git", "pull"], check=True)

# Reset vocab files if Kaggle has local changes from prior session
subprocess.run(["git", "checkout", "--", "artifacts/vocab_de.pt", "artifacts/vocab_en.pt"], check=False)

print("Pre-flight complete")
```

### Step 1: Ablation runs

Each ablation uses v5 architecture (small + fast). Run config:

```python
# Base v5 config — copy this and override per ablation
BASE = {
    "d_model": 256, "N": 3, "num_heads": 8, "d_ff": 512, "dropout": 0.1,
    "max_len": 128, "batch_size": 128, "num_epochs": 20, "warmup_steps": 4000,
    "label_smooth": 0.1, "betas": (0.9, 0.98), "eps": 1e-9, "grad_clip": 1.0,
    "seed": 25014, "eval_every": 2, "use_amp": True, "early_stop_patience": 5,
    "scheduler": "noam", "use_scaling": True, "pos_encoding": "sinusoidal",
    "use_expanded": False, "wandb_project": "da6401-a3-transformer",
    "wandb_group": "ablations", "artifacts_dir": "artifacts",
}
```

**Ablation 2.1 (Fixed LR vs Noam):** override `"scheduler": "fixed"`, `"run_name": "abl-2.1-fixed-lr"`, `"checkpoint": "/kaggle/working/abl_2.1.pt"`

**Ablation 2.2 (No scaling):** override `"use_scaling": False`, `"run_name": "abl-2.2-no-scaling"`, `"checkpoint": "/kaggle/working/abl_2.2.pt"`

**Ablation 2.4 (Learned PE):** override `"pos_encoding": "learned"`, `"run_name": "abl-2.4-learned-pe"`, `"checkpoint": "/kaggle/working/abl_2.4.pt"`

**Ablation 2.5 (No label smoothing):** override `"label_smooth": 0.0`, `"run_name": "abl-2.5-no-smoothing"`, `"checkpoint": "/kaggle/working/abl_2.5.pt"`

**For section 2.3 (Attention Rollout):** no new training run needed. Load the v5 OR v7 checkpoint and extract attention maps from a few example sentences. Visualize.

### Step 2: Report writing

Use `wandb_workspaces.reports.v2` API to publish programmatically. Outline per section:
- 1-paragraph hypothesis
- Loss curves (train + val)
- BLEU curves (val_bleu)
- Test BLEU table (main vs ablation)
- 2-3 paragraph observation + commentary

Friend's published report (for format reference, not content):
`https://api.wandb.ai/links/jaydeep316-i/szdhqk77`

## Recovery procedures (if something breaks)

### If Gradescope autograder fails tomorrow

Worst case: someone resets it or the Drive download fails. Recovery:

```bash
cd ~/da6401_assignment_3
# Restore v7 checkpoint (the one currently on Drive)
# Upload backups/transformer_main_v7_BACKUP.pt to Drive as new version
# OR fall back to v5:
#   Upload backups/transformer_main_v5_BACKUP.pt to Drive
#   Revert model.py defaults to d_model=256, N=3, d_ff=512, dropout=0.1
#   git commit + push
#   Resubmit
```

### If you want to switch v7 → v5 tomorrow

1. Upload `backups/transformer_main_v5_BACKUP.pt` to Drive (replace v7)
2. Revert `model.py` defaults:
   ```bash
   cd ~/da6401_assignment_3
   python3 << 'PYEOF'
   from pathlib import Path
   p = Path("model.py"); s = p.read_text()
   s = s.replace(
       "d_model:   int   = 512,\n        N:         int   = 6,\n        num_heads: int   = 8,\n        d_ff:      int   = 2048,\n        dropout:   float = 0.15,",
       "d_model:   int   = 256,\n        N:         int   = 3,\n        num_heads: int   = 8,\n        d_ff:      int   = 512,\n        dropout:   float = 0.1,"
   )
   p.write_text(s)
   print("Reverted to v5 defaults")
   PYEOF
   git add model.py
   git commit -m "Revert to v5 architecture"
   git push origin main
   ```
3. Resubmit Gradescope → expect 36.88 BLEU again

## Key files in repo

- `model.py` — Transformer (currently v7 defaults)
- `train.py` — supports ablation flags: `scheduler`, `use_scaling`, `pos_encoding`, `label_smooth`
- `dataset.py` — lowercase tokenizer + Multi30k loader
- `lr_scheduler.py` — Noam scheduler
- `artifacts/vocab_de.pt` (7851 lowercase tokens)
- `artifacts/vocab_en.pt` (5892 lowercase tokens)
- `backups/` — v5 (90MB) + v7 (614MB) checkpoints (LOCAL ONLY, gitignored)

## Resources

- Repo: `https://github.com/DA25M014/da6401_assignment_3`
- Drive checkpoint file ID: `15e-O7Ji4kMnLrmaTBY5hmjnW6xpuJKGS`
- Kaggle notebook: `da6401-a3-transformer-train`
- W&B project: `https://wandb.ai/da25m014-iitm/da6401-a3-transformer`
- Friend's repo (for diff comparison only, not copying): `https://github.com/makwana-jaydeep/da6401_assignment_3`

---

## Quick mental model for tomorrow

You have **50/50 from the autograder, locked.** The only thing left is to:
1. Run 4 ablations (~2 hours wall time on Kaggle, you can multitask)
2. Visualize attention heads from v7 (~30 min)
3. Write up the W&B report with observations (~3 hours)

Total: ~5 hours of focused work. Probably do it in 2 sessions across tomorrow + Monday morning.

**Final target: 50 (auto) + ~45+ (report) = 95+/100.**
