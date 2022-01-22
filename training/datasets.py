import random
from itertools import accumulate
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


class ToxicTextsDataset(Dataset):
    @classmethod
    def from_file(cls, path, vocab, epoch_size, bpe_dropout=0,
                  max_positions=1024, balanced=False):
        data = pd.read_csv(path)

        data['comment_text'] = data['comment_text']

        cat_weights = {'threat': 1,
                       'identity_attack': 1,
                       'insult': 1,
                       'severe_toxicity': 1,
                       'sexual_explicit': 1,
                       'obscene': 1}
        data['target'] = sum([data[c] * w for c, w in cat_weights.items()])

        return cls(data['comment_text'].tolist(), data['target'].tolist(),
                   vocab, epoch_size, bpe_dropout, max_positions, balanced)

    def __init__(self, texts, targets, vocab, epoch_size, bpe_dropout=0,
                 max_positions=1024, balanced=False):
        super().__init__()

        self.data = list(zip(texts, targets))
        self.vocab = vocab
        self.epoch_size = epoch_size
        self.bpe_dropout = bpe_dropout
        self.max_positions = max_positions
        self._head_size = int(0.25 * self.max_positions)
        self._tail_size = self.max_positions - self._head_size

        if balanced:
            rounded_targets = [round(t, 1) for t in targets]
            counter = Counter(rounded_targets)
            self.cum_sampling_weights = list(accumulate([1 / counter[t] for t in rounded_targets]))
        else:
            self.cum_sampling_weights = list(range(len(self.data)))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        text, target = random.choices(self.data, cum_weights=self.cum_sampling_weights, k=1)[0]
        tokens = self.vocab.text2ids(text, self.bpe_dropout) + [self.vocab.eos_id]
        if len(tokens) > self.max_positions:
            tokens = tokens[:self._head_size] + tokens[-self._tail_size:]
        tokens = torch.tensor(tokens, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.float)

        return tokens, target


class RangingToxicTextsDataset(Dataset):
    @classmethod
    def from_file(cls, path, vocab, bpe_dropout=0, max_positions=1024):
        data = pd.read_csv(path)
        less_toxic_texts = data['less_toxic']
        more_toxic_texts = data['more_toxic']

        return cls(less_toxic_texts.tolist(), more_toxic_texts.tolist(),
                   vocab, bpe_dropout, max_positions)

    def __init__(self, less_toxic_texts, more_toxic_texts, vocab, bpe_dropout=0, max_positions=1024):
        super().__init__()

        self.data = list(zip(less_toxic_texts, more_toxic_texts))
        self.vocab = vocab
        self.bpe_dropout = bpe_dropout
        self.max_positions = max_positions
        self._head_size = int(0.25 * self.max_positions)
        self._tail_size = self.max_positions - self._head_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        less_toxic_text, more_toxic_text = self.data[idx]

        less_toxic_tokens = self.vocab.text2ids(less_toxic_text, self.bpe_dropout) + [self.vocab.eos_id]
        if len(less_toxic_tokens) > self.max_positions:
            less_toxic_tokens = less_toxic_tokens[:self._head_size] + less_toxic_tokens[-self._tail_size:]
        more_toxic_tokens = self.vocab.text2ids(more_toxic_text, self.bpe_dropout) + [self.vocab.eos_id]
        if len(more_toxic_tokens) > self.max_positions:
            more_toxic_tokens = more_toxic_tokens[:self._head_size] + more_toxic_tokens[-self._tail_size:]
        less_toxic_tokens = torch.tensor(less_toxic_tokens, dtype=torch.long)
        more_toxic_tokens = torch.tensor(more_toxic_tokens, dtype=torch.long)

        return less_toxic_tokens, more_toxic_tokens
