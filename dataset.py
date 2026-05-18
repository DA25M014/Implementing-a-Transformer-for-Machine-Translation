'''
dataset.py -- Multi30k loader, spaCy tokenization, custom Vocab.

No torchtext (deprecated). Vocab is a thin Python dict with explicit
special-token ordering: <unk>=0, <pad>=1, <sos>=2, <eos>=3.

Pad index is hard-coded to 1 because model.py\'s make_src_mask and
make_tgt_mask default to pad_idx=1.
'''

import os
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

import spacy
from datasets import load_dataset


# ---------------------------------------------------------------------
#  VOCAB
# ---------------------------------------------------------------------

class Vocab:
    '''
    Minimal vocabulary class. Stores stoi (string -> int) and itos (int -> string).

    Special tokens are baked in at fixed positions:
        <unk>=0, <pad>=1, <sos>=2, <eos>=3
    '''

    SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
    UNK_IDX  = 0
    PAD_IDX  = 1
    SOS_IDX  = 2
    EOS_IDX  = 3

    def __init__(self, stoi: dict, itos: list):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build_from_iterator(cls, token_iter, min_freq: int = 2) -> "Vocab":
        counter = Counter()
        for tokens in token_iter:
            counter.update(tokens)

        # Special tokens occupy fixed slots regardless of frequency.
        itos = list(cls.SPECIALS)
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < min_freq:
                continue
            if tok in cls.SPECIALS:          # avoid double-inserting specials
                continue
            itos.append(tok)

        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi, itos)

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.UNK_IDX)

    def encode(self, tokens: list[str], add_specials: bool = True) -> list[int]:
        '''Convert tokens to ids. With add_specials, wraps in <sos> ... <eos>.'''
        ids = [self.stoi.get(t, self.UNK_IDX) for t in tokens]
        if add_specials:
            ids = [self.SOS_IDX] + ids + [self.EOS_IDX]
        return ids

    def decode(self, ids: list[int], strip_specials: bool = True) -> list[str]:
        '''Convert ids back to tokens. Optionally drop all special tokens.'''
        tokens = [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]
        if strip_specials:
            tokens = [t for t in tokens if t not in self.SPECIALS]
        return tokens

    def save(self, path: str | Path) -> None:
        torch.save({"stoi": self.stoi, "itos": self.itos}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        blob = torch.load(path, weights_only=False)
        return cls(blob["stoi"], blob["itos"])


# ---------------------------------------------------------------------
#  TOKENIZER HELPERS
# ---------------------------------------------------------------------

def _make_spacy_tokenizer(model_name: str) -> Callable[[str], list[str]]:
    '''Returns a function that maps a raw string to a list of token strings.'''
    nlp = spacy.load(model_name, disable=["parser", "ner", "tagger", "lemmatizer"])

    def tokenize(text: str) -> list[str]:
        return [tok.text.lower() for tok in nlp(text.strip()) if not tok.is_space]

    return tokenize


# ---------------------------------------------------------------------
#  MULTI30K DATASET
# ---------------------------------------------------------------------

class Multi30kDataset(Dataset):
    '''
    Multi30k German->English translation dataset, wrapping the
    bentrevett/multi30k HuggingFace release.

    Acts as a PyTorch Dataset: __getitem__ returns (src_ids, tgt_ids)
    as 1D LongTensors WITH <sos>/<eos> wrappers but WITHOUT padding.
    Padding is the collate function\'s job (see collate_batch).

    Vocabularies are built once from the TRAIN split and saved to disk.
    Validation and test splits reuse the same vocabs.
    '''

    HF_NAME = "bentrevett/multi30k"

    def __init__(
        self,
        split: str = "train",
        artifacts_dir: str | Path = "artifacts",
        max_len: int = 128,
        use_expanded: bool = False,
    ):
        assert split in {"train", "validation", "test"}, f"bad split {split}"
        self.split = split
        self.artifacts_dir = Path(artifacts_dir)
        self.max_len = max_len
        self.use_expanded = use_expanded

        self.tokenize_de = _make_spacy_tokenizer("de_core_news_sm")
        self.tokenize_en = _make_spacy_tokenizer("en_core_web_sm")

        if use_expanded:
            # Load pre-built multi-reference parquet.
            import pandas as pd
            split_file = {"train": "expanded_train.parquet",
                          "validation": "expanded_val.parquet",
                          "test": "expanded_test.parquet"}[split]
            df = pd.read_parquet(self.artifacts_dir / split_file)
            # Build a list of {"de": ..., "en": ...} matching HF interface.
            self._hf_split = [{"de": r.de, "en": r.en} for r in df.itertuples(index=False)]
        else:
            self._hf_split = load_dataset(self.HF_NAME, split=split)

        # Vocabs may not exist yet (first build call).
        self.src_vocab: Vocab | None = None
        self.tgt_vocab: Vocab | None = None
        self._try_load_vocabs()

    # -- public API ---------------------------------------------------

    def build_vocab(self, min_freq: int = 2) -> tuple[Vocab, Vocab]:
        '''
        Build src (de) and tgt (en) vocabularies from THIS split.

        Should be called on the TRAIN split only; vocabs are then saved
        to artifacts/ and reused by val/test instances via _try_load_vocabs.
        '''
        if self.split != "train":
            raise RuntimeError(
                f"build_vocab must be called on train split, got {self.split!r}"
            )

        de_iter = (self.tokenize_de(ex["de"]) for ex in self._hf_split)
        en_iter = (self.tokenize_en(ex["en"]) for ex in self._hf_split)

        self.src_vocab = Vocab.build_from_iterator(de_iter, min_freq=min_freq)
        self.tgt_vocab = Vocab.build_from_iterator(en_iter, min_freq=min_freq)

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.src_vocab.save(self.artifacts_dir / "vocab_de.pt")
        self.tgt_vocab.save(self.artifacts_dir / "vocab_en.pt")

        return self.src_vocab, self.tgt_vocab

    def _try_load_vocabs(self) -> None:
        de_path = self.artifacts_dir / "vocab_de.pt"
        en_path = self.artifacts_dir / "vocab_en.pt"
        if de_path.exists() and en_path.exists():
            self.src_vocab = Vocab.load(de_path)
            self.tgt_vocab = Vocab.load(en_path)

    # -- PyTorch Dataset interface ------------------------------------

    def __len__(self) -> int:
        return len(self._hf_split)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Vocab not built/loaded. Call build_vocab() on a train instance first."
            )

        ex = self._hf_split[idx]
        de_tokens = self.tokenize_de(ex["de"])[: self.max_len - 2]   # leave room for <sos>/<eos>
        en_tokens = self.tokenize_en(ex["en"])[: self.max_len - 2]

        src_ids = self.src_vocab.encode(de_tokens, add_specials=True)
        tgt_ids = self.tgt_vocab.encode(en_tokens, add_specials=True)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


# ---------------------------------------------------------------------
#  COLLATE FUNCTION
# ---------------------------------------------------------------------

def collate_batch(batch, pad_idx: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Pad a batch of (src_ids, tgt_ids) variable-length sequences to the
    longest in the batch. Returns:
        src : [B, max_src_len]  long
        tgt : [B, max_tgt_len]  long
    Use with DataLoader: collate_fn=collate_batch
    '''
    srcs, tgts = zip(*batch)

    max_src = max(s.size(0) for s in srcs)
    max_tgt = max(t.size(0) for t in tgts)

    src_padded = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, : s.size(0)] = s
        tgt_padded[i, : t.size(0)] = t

    return src_padded, tgt_padded
