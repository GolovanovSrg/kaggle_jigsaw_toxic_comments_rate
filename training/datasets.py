import re

import pandas as pd
import torch
from bs4 import BeautifulSoup
from torch.utils.data import Dataset


def _text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis

    text - Text piece to be cleaned.
    '''

    template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    text = template.sub(r'', text)

    soup = BeautifulSoup(text, 'lxml')  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile('['
                            u'\U0001F600-\U0001F64F'  # emoticons
                            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                            u'\U0001F680-\U0001F6FF'  # transport & map symbols
                            u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                            u'\U00002702-\U000027B0'
                            u'\U000024C2-\U0001F251'
                            ']+', flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r'[^a-zA-Z\d]', ' ', text)  # Remove special Charecters
    text = re.sub(' +', ' ', text)  # Remove Extra Spaces
    text = text.strip()  # Remove spaces at the beginning and at the end of string

    return text


class ToxicTextsDataset(Dataset):
    @classmethod
    def from_file(cls, path, vocab, bpe_dropout=0, max_positions=1024, balanced=False):
        cat_weights = {'obscene': 0.16,
                       'toxic': 0.32,
                       'threat': 1.5,
                       'insult': 0.64,
                       'severe_toxic': 1.5,
                       'identity_hate': 1.5}

        data = pd.read_csv(path)
        data['target'] = sum([data[c] * w for c, w in cat_weights.items()])

        if balanced:
            n_toxic = (data['target'] >= 0.1).sum()
            data = pd.concat([data[data['target'] >= 0.1], data[data['target'] == 0].sample(n=n_toxic, random_state=0)])

        texts = data['comment_text'].apply(_text_cleaning)
        targets = data['target']

        return cls(texts.tolist(), targets.tolist(),
                   vocab, bpe_dropout, max_positions)

    def __init__(self, texts, targets, vocab, bpe_dropout=0, max_positions=1024):
        super().__init__()

        self.vocab = vocab
        self.bpe_dropout = bpe_dropout
        self.max_positions = max_positions
        self.data = list(zip(texts, targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, target = self.data[idx]
        tokens = self.vocab.text2ids(text, self.bpe_dropout)[:self.max_positions-1] + [self.vocab.eos_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.float)

        return tokens, target


class RangingToxicTextsDataset(Dataset):
    @classmethod
    def from_file(cls, path, vocab, bpe_dropout=0, max_positions=1024):
        data = pd.read_csv(path)
        less_toxic_texts = data['less_toxic'].apply(_text_cleaning)
        more_toxic_texts = data['more_toxic'].apply(_text_cleaning)

        return cls(less_toxic_texts.tolist(), more_toxic_texts.tolist(),
                   vocab, bpe_dropout, max_positions)

    def __init__(self, less_toxic_texts, more_toxic_texts, vocab, bpe_dropout=0, max_positions=1024):
        super().__init__()

        self.vocab = vocab
        self.bpe_dropout = bpe_dropout
        self.max_positions = max_positions
        self.data = list(zip(less_toxic_texts, more_toxic_texts))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        less_toxic_text, more_toxic_text = self.data[idx]
        less_toxic_tokens = self.vocab.text2ids(less_toxic_text, self.bpe_dropout)[:self.max_positions-1] + [self.vocab.eos_id]
        more_toxic_tokens = self.vocab.text2ids(more_toxic_text, self.bpe_dropout)[:self.max_positions-1] + [self.vocab.eos_id]
        less_toxic_tokens = torch.tensor(less_toxic_tokens, dtype=torch.long)
        more_toxic_tokens = torch.tensor(more_toxic_tokens, dtype=torch.long)

        return less_toxic_tokens, more_toxic_tokens
