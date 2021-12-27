import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEmbedding(nn.Embedding):
    def __init__(self, embedding_dim, n_embeddings):
        n_embeddings += 1  # 0 is the padding
        super().__init__(n_embeddings, embedding_dim, padding_idx=0)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=0.01)


class CombinedEmbedding(nn.Module):
    def __init__(self, n_embeddings, n_pos_embeddings, embedding_dim, dropout=0, padding_idx=None):
        super().__init__()

        self.tok_padding_idx = padding_idx
        self.pos_padding_idx = 0

        self.tok_embedding = nn.Embedding(n_embeddings, embedding_dim, padding_idx=self.tok_padding_idx)
        self.pos_embedding = LearnablePositionalEmbedding(embedding_dim, n_pos_embeddings)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_embedding.weight, std=0.02)

    def forward(self, x):
        padding_mask = x[:, :].eq(self.tok_padding_idx)
        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, self.pos_padding_idx)

        x = self.tok_embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        return x, padding_mask


class SelfMultiheadAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, size, device):
        if not hasattr(cls, '_future_mask') or cls._future_mask.device != device or cls._future_mask.shape < size:
            cls._future_mask = torch.triu(torch.ones(size[0], size[1], dtype=torch.uint8, device=device), 1).bool()
        mask = cls._future_mask[:size[0], :size[1]]

        return mask

    def __init__(self, n_features, n_heads, dropout, future_mask=True):
        super().__init__()

        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.future_mask =future_mask
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x, is_key=False):
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v, padding_mask):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)

        if self.future_mask:
            future_mask = self._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0)
            w.masked_fill_(future_mask, float('-inf'))

        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, x, padding_mask):
        query, key, value = self.qkv_proj(x).split(self.n_features, dim=-1)

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value, padding_mask)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, in_features, middle_features, dropout):
        super().__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.activatin = nn.GELU()
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.zeros_(self.layer_1.bias)
        nn.init.normal_(self.layer_2.weight, std=0.02)
        nn.init.zeros_(self.layer_2.bias)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.activatin(x)
        x = self.dropout(x)
        x = self.layer_2(x)

        return x


class Adapter(nn.Module):
    """
    https://arxiv.org/pdf/1902.00751.pdf
    """

    def __init__(self, in_features, middle_features):
        super().__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.activation = nn.GELU()
        self.layer_2 = nn.Linear(middle_features, in_features)

        self._init_weights()

    def _init_weights(self) :
        nn.init.normal_(self.layer_1.weight, std=1e-6)
        nn.init.zeros_(self.layer_1.bias)
        nn.init.normal_(self.layer_2.weight, std=1e-6)
        nn.init.zeros_(self.layer_2.bias)

    def forward(self, x):
        residual = x

        x = self.layer_1(x)
        x = self.activation(x)
        x = self.layer_2(x)

        x = residual + x

        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout=0, attn_dropout=0, ff_dropout=0,
                 future_mask=True, adapters=False, adapters_size=None):
        super().__init__()

        if adapters_size is None:
            adapters_size = n_features

        self.attn_norm = nn.LayerNorm(n_features)
        self.attn = SelfMultiheadAttention(n_features, n_heads, attn_dropout, future_mask)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_adapter = Adapter(n_features, adapters_size) if adapters else None

        self.ff_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_adapter = Adapter(n_features, adapters_size) if adapters else None

    def _process_attn(self, x, padding_mask):
        residual = x

        x = self.attn_norm(x)
        x = self.attn(x, padding_mask)
        x = self.attn_dropout(x)

        if self.attn_adapter is not None:
            x = self.attn_adapter(x)

        x = residual + x

        return x

    def _process_ff(self, x):
        residual = x

        x = self.ff_norm(x)
        x = self.ff(x)
        x = self.ff_dropout(x)

        if self.ff_adapter is not None:
            x = self.ff_adapter(x)

        x = residual + x

        return x

    def forward(self, x, padding_mask):
        x = self._process_attn(x, padding_mask)
        x = self._process_ff(x)

        return x


class Transformer(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embedding_dim, padding_idx, n_heads,
                 dropout=0, embedding_dropout=0, attn_dropout=0, ff_dropout=0, adapters=False,
                 adapters_size=None):
        super().__init__()

        self.embedding = CombinedEmbedding(n_embeddings=n_embeddings,
                                           n_pos_embeddings=n_pos_embeddings,
                                           embedding_dim=embedding_dim,
                                           dropout=embedding_dropout,
                                           padding_idx=padding_idx)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerBlock(n_features=embedding_dim,
                                     n_heads=n_heads,
                                     dropout=dropout,
                                     attn_dropout=attn_dropout,
                                     ff_dropout=ff_dropout,
                                     future_mask=True,
                                     adapters=adapters,
                                     adapters_size=adapters_size)
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(embedding_dim)

        if adapters:
            self.requires_grad_(False)
            for layer in self.layers:
                layer.attn_norm.requires_grad_(True)
                layer.attn_adapter.requires_grad_(True)
                layer.ff_norm.requires_grad_(True)
                layer.ff_adapter.requires_grad_(True)
            self.final_norm.requires_grad_(True)

    def forward(self, x, return_lm_logits=False):
        x, padding_mask = self.embedding(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)

        if return_lm_logits:
            lm_logits = F.linear(x, self.embedding.tok_embedding.weight)
            return x, lm_logits, padding_mask
        return x, padding_mask
