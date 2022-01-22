import torch
import torch.nn as nn

from gpt2 import Transformer


class AdaptersModel(nn.Module):
    def __init__(self, n_targets, n_layers, n_embeddings, n_pos_embeddings, embedding_dim, padding_idx,
                 n_heads, dropout=0, embedding_dropout=0, attn_dropout=0, ff_dropout=0, drop_path=0,
                 adapters_dropout=0, adapters=False, adapters_size=None):
        super().__init__()

        self.padding_idx = padding_idx

        self.encoder = Transformer(n_layers=n_layers,
                                   n_embeddings=n_embeddings,
                                   n_pos_embeddings=n_pos_embeddings,
                                   embedding_dim=embedding_dim, 
                                   padding_idx=padding_idx,
                                   n_heads=n_heads,
                                   dropout=dropout,
                                   embedding_dropout=embedding_dropout,
                                   attn_dropout=attn_dropout,
                                   ff_dropout=ff_dropout,
                                   drop_path=drop_path,
                                   adapters_dropout=adapters_dropout,
                                   adapters=adapters,
                                   adapters_size=adapters_size)

        self.target_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                          nn.GELU(),
                                          nn.Linear(embedding_dim, n_targets))

    def forward(self, x):
        orig = x
        (x, lm_logits), padding_mask = self.encoder(x, return_lm_logits=True)
        lengths = (~padding_mask).long().sum(dim=-1)
        lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x.gather(1, lengths-1).squeeze(1)
        out = self.target_layer(x).view(-1)

        return out, lm_logits, padding_mask


class PromptModel(nn.Module):
    def __init__(self, n_targets, n_layers, n_embeddings, n_pos_embeddings, embedding_dim, padding_idx,
                 n_heads, left_prompt_size, right_prompt_size, dropout=0, embedding_dropout=0,
                 attn_dropout=0, ff_dropout=0, drop_path=0):
        super().__init__()

        self.padding_idx = padding_idx

        self.encoder = Transformer(n_layers=n_layers,
                                   n_embeddings=n_embeddings,
                                   n_pos_embeddings=n_pos_embeddings,
                                   embedding_dim=embedding_dim, 
                                   padding_idx=padding_idx,
                                   n_heads=n_heads,
                                   dropout=dropout,
                                   embedding_dropout=embedding_dropout,
                                   attn_dropout=attn_dropout,
                                   ff_dropout=ff_dropout,
                                   drop_path=drop_path)
        self.encoder.requires_grad_(False)

        self.target_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                          nn.GELU(),
                                          nn.Linear(embedding_dim, n_targets))

        self.left_prompt_size = left_prompt_size
        self.right_prompt_size = right_prompt_size
        self.register_buffer('left_prompt_padding', torch.zeros(1, left_prompt_size, dtype=torch.bool))
        self.register_buffer('right_prompt_padding', torch.zeros(1, right_prompt_size, dtype=torch.bool))
        prompt = self.encoder.embedding.tok_embedding.weight.data.mean(dim=0)
        prompt = prompt[None, None].repeat(1, left_prompt_size + right_prompt_size, 1)
        self.prompt = nn.Parameter(prompt)
        self.prompt_nn = nn.LSTM(embedding_dim, embedding_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.prompt_proj = nn.Linear(2 * embedding_dim, embedding_dim)


    def forward(self, x):
        prompt = self.prompt_proj(self.prompt_nn(self.prompt)[0])
        left_prompt = prompt[:, :self.left_prompt_size].repeat(x.shape[0], 1, 1)
        right_prompt = prompt[:, -self.right_prompt_size:].repeat(x.shape[0], 1, 1)

        tok_embs, padding_mask = self.encoder.embedding.get_token_embeddings(x)
        tok_embs = torch.cat([left_prompt, tok_embs, right_prompt], dim=1)
        padding_mask = torch.cat([self.left_prompt_padding.repeat(x.shape[0], 1), padding_mask,
                                  self.right_prompt_padding.repeat(x.shape[0], 1)], dim=1)

        pos_embs = self.encoder.embedding.get_position_embeddings(padding_mask)
        embs = self.encoder.embedding.dropout(tok_embs + pos_embs)

        out, lm_logits = self.encoder.transformer_layers(embs, padding_mask, return_lm_logits=True)
        lm_logits = lm_logits[:, self.left_prompt_size:-self.right_prompt_size]
        padding_mask = padding_mask[:, self.left_prompt_size:-self.right_prompt_size]

        out = self.target_layer(out[:, -1]).view(-1)

        return out, lm_logits, padding_mask


# class Predictor:
#     def __init__(self, model, vocab, device=None):
#         if device is None:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         device = torch.device(device)

#         self._model = model.to(device)
#         self._vocab = vocab
#         self._device = device

#     @torch.no_grad()
#     def __call__(self, texts):
#         self._model.eval()

#         tokens = [torch.tensor(self._vocab.string2ids(text)[:self._model.n_pos_embeddings-1]+[self._vocab.eos_id], dtype=torch.long) for text in texts]
#         tokens = pad_sequence(tokens, batch_first=True, padding_value=self._model.padding_idx)

#         tokens = tokens.to(self._device)
#         predictions = self._model.predict(tokens)

#         return predictions.cpu().numpy()
