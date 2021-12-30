import torch.nn as nn

from gpt2 import Transformer


class RegressionModel(nn.Module):
    def __init__(self, n_targets, n_layers, n_embeddings, n_pos_embeddings, embedding_dim, padding_idx,
                 n_heads, dropout=0, embedding_dropout=0, attn_dropout=0, ff_dropout=0, adapters=False,
                 adapters_size=None):
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
                                   adapters=adapters,
                                   adapters_size=adapters_size)
        self.target_layer = nn.Linear(embedding_dim, n_targets)

    def forward(self, x):
        x, lm_logits, padding_mask = self.encoder(x, return_lm_logits=True)
        lengths = (~padding_mask).long().sum(dim=-1)
        lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x.gather(1, lengths-1).squeeze(1)
        out = self.target_layer(x).view(-1)

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
