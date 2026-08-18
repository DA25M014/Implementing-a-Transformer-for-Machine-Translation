# Implementing a Transformer for Machine Translation

## Transformer for German-English Machine Translation
> **Roll No:** DA25M014 · **Course:** DA6401 Deep Learning · **IIT Madras**
---


### 🔗 Quick Links

|  |  |
| --- | --- |
| 📊 **W&B Report** | [View Report](https://api.wandb.ai/links/da25m014-iitm/0pm0zad2) |
| 💻 **GitHub Repo** | [View Code](https://github.com/DA25M014/da6401_assignment_3) |

---


### 📌 About

A **from-scratch PyTorch implementation of the Transformer architecture** from *Attention Is All You Need* (Vaswani et al., 2017), trained for German-to-English Neural Machine Translation on the Multi30k dataset (29,000 sentence pairs). Each design choice of the original paper is verified through controlled single-variable ablations against the unmodified baseline.

| Component | Detail |
| --- | --- |
| Architecture | 51M parameters · d_model=512 · N=6 layers · 8 heads · d_ff=2048 |
| Vocabulary | German 7,851 · English 5,892 · spaCy tokenization · lowercase |
| Training | 20 epochs · batch 128 · Noam scheduler (2k warmup) · label smoothing ε=0.1 |
| Hardware | Kaggle T4 ×2 GPU · mixed-precision (bf16) |
| **Test BLEU (local)** | **32.78** |
| **Autograder BLEU (Gradescope hidden test set)** | **35.23** |

---


### 🧪 W&B Experiments (Ablation Study)

| # | Section | Variable Changed | Δ Test BLEU | Finding |
| --- | --- | --- | --- | --- |
| 2.1 | Noam Scheduler | Fixed LR = 1e-4 (no warmup) | -12.76 | Without warmup, peak LR cannot be reached safely; convergence stalls |
| 2.2 | Scaling Factor | Remove 1/√dₖ | **-26.46** | Training collapses at epoch 5; gradient norms spike then vanish |
| 2.3 | Attention Rollout | (visualization only) | — | 8 heads specialize: predecessor (H2), modifier-head (H4, H5), long-range (H3, H7, H8); partial redundancy in attention-sink cluster |
| 2.4 | Positional Encoding | Learned `nn.Embedding` instead of sinusoidal | -1.23 | Sinusoidal wins despite zero parameters; learned PE cannot extrapolate beyond train length |
| 2.5 | Label Smoothing | ε = 0.0 (standard CE) | -1.10 | Confidence rises faster but model overfits; calibration suffers at inference |

---


### 📂 Project Structure

```
├── artifacts/
│   ├── vocab_de.pt              # German vocabulary (7,851 tokens)
│   └── vocab_en.pt              # English vocabulary (5,892 tokens)
├── tests/
│   ├── test_attention.py        # MHA + scaled dot-product attention
│   ├── test_positional_encoding.py
│   ├── test_noam_scheduler.py
│   ├── test_transformer_init.py
│   ├── test_transformer_smoke.py
│   ├── test_infer.py            # End-to-end infer() autograder contract
│   ├── test_dataset.py
│   ├── test_layers_smoke.py
│   └── test_train_pipeline.py
├── model.py                     # Transformer + MHA + PE + Encoder/Decoder stacks
├── dataset.py                   # Multi30k loading + spaCy tokenization + Vocab
├── train.py                     # Training pipeline + ablation flags + W&B logging
├── lr_scheduler.py              # Noam scheduler
└── requirements.txt

```

---


### 🚀 Usage

#### Inference (Gradescope autograder contract)

```python
from model import Transformer

model = Transformer().to(device)
model.eval()
english_sentence = model.infer(german_sentence)
```

The `Transformer()` constructor downloads pretrained weights from Google Drive via `gdown`, loads vocab + spaCy tokenizers, and instantiates the model — all inside `__init__`.

#### Training (full reproduction)

```bash
# Default v7 configuration (51M params, 20 epochs)
python train.py

# Ablation 2.1: Fixed learning rate (no Noam warmup)
python train.py --scheduler fixed --fixed-lr 1e-4 --warmup-steps 0

# Ablation 2.2: Remove 1/sqrt(d_k) scaling
python train.py --no-scaling

# Ablation 2.4: Learned positional embedding
python train.py --pos-encoding learned

# Ablation 2.5: No label smoothing
python train.py --label-smooth 0.0
```

#### Run tests

```bash
pytest tests/ -v
```

---


### 📦 Dependencies

```
torch
numpy
matplotlib
scikit-learn
wandb
datasets
spacy
sacrebleu
tqdm
gdown

```

Plus the spaCy language models (auto-downloaded on first run):

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---


**DA25M014** · M.Tech · IIT Madras

[GitHub](https://github.com/DA25M014/da6401_assignment_3) · [W&B Report](https://api.wandb.ai/links/da25m014-iitm/0pm0zad2)

---


DA6401 · Deep Learning · IIT Madras · 2025–26
