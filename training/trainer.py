import os
import random

import torch as t
import torch.nn.functional as F
import torchsort
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import AvgValue, add_weight_decay


class Trainer:
    @staticmethod
    def _set_seed(seed=0):
        random.seed(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False
        t.manual_seed(seed)

    @staticmethod
    def _label_smooth_loss(inputs, targets, padding_mask, smoothing):
        weight = t.full_like(inputs, smoothing / (inputs.shape[-1] - 1))
        weight.scatter_(-1, targets.unsqueeze(-1), (1 - smoothing))
        mask = 1 - padding_mask[..., None].float()
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = (-weight * mask * log_probs).sum(dim=-1).sum() / mask.sum()

        return loss

    @staticmethod
    def _margin_rank_loss(preds, targets, margin):
        dists = preds.view(-1, 1) - preds.view(1, -1)
        mult = 2 * (targets.view(-1, 1) >= targets.view(1, -1)).float() - 1
        loss = (margin - mult * dists).clamp(min=0).triu(diagonal=1).mean()

        return loss

    # @staticmethod
    # def _soft_rank_loss(pred, target):
    #     pred_ranks = [torchsort.soft_rank(pred.view(1, -1), regularization='l2', regularization_strength=r)
    #                   for r in [1, 0.5, 0.1, 0.05, 0.01]]

    #     tmp = target.view(1, -1).argsort()
    #     target_ranks = t.zeros_like(tmp)
    #     target_ranks[:, tmp] = t.arange(1, target_ranks.shape[-1] + 1, device=target_ranks.device)

    #     n = target_ranks.shape[-1]
    #     mse_diffs = sum([(target_ranks - p).pow(2).sum() for p in pred_ranks], 0) / len(pred_ranks)
    #     upper = 6 * mse_diffs
    #     down = n * (n ** 2 - 1.0)
    #     loss = upper / down - 1

    #     return loss

    def __init__(self, model, optimizer_params={}, loss_params={}, clip_grad_norm=None,
                 tb_dir=None, n_chunks=1, device=None, n_jobs=1, seed=0):
        self._set_seed(seed)

        if device is None:
            device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        else:
            device = t.device(device)

        self.model = model.to(device)

        margin = loss_params.get('margin', 1)
        lm_weight = loss_params.get('lm_weight', 0)
        smoothing = loss_params.get('smoothing', 0)
        self.trg_criterion = lambda *args: self._margin_rank_loss(*args, margin=margin)
        self.lm_criterian = lambda *args: t.tensor([0], dtype=t.float, device=device, requires_grad=True)
        if lm_weight > 0:
            self.lm_criterian = lambda *args: lm_weight * self._label_smooth_loss(*args, smoothing=smoothing)

        warmup = optimizer_params.pop('warmup', 0)
        gamma = optimizer_params.pop('gamma', 1)
        scheduler_func = lambda i: min(i / (warmup+1), gamma ** (i-warmup))

        params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        params = add_weight_decay(params, optimizer_params.get('weight_decay', 0))
        self.optimizer = t.optim.AdamW(params, **optimizer_params)
        self.scheduler = LambdaLR(self.optimizer, scheduler_func)
        self.clip_grad_norm = clip_grad_norm

        self.writer = SummaryWriter(log_dir=tb_dir)

        self.n_chunks = n_chunks
        self.device = device
        self.n_jobs = n_jobs

        self.last_epoch = 0

    def _train_epoch(self, train_dataloader):
        tqdm_train_dataloader = tqdm(train_dataloader, desc=f'Train, epoch #{self.last_epoch}')
        self.model.train()

        trg_loss, lm_loss = AvgValue(), AvgValue()
        for tokens, targets in tqdm_train_dataloader:
            self.optimizer.zero_grad()

            tokens, targets = tokens.to(self.device), targets.to(self.device)
            chunks = zip(tokens.chunk(self.n_chunks), targets.chunk(self.n_chunks))
            for tokens_chunk, targets_chunk in chunks:
                preds, lm_logits, padding_mask = self.model(tokens_chunk)
                chunk_trg_loss = self.trg_criterion(preds, targets_chunk)
                chunk_lm_loss = self.lm_criterian(lm_logits[:, :-1], tokens_chunk[:, 1:], padding_mask[:, 1:])
                chunk_loss = (chunk_trg_loss + chunk_lm_loss) / self.n_chunks
                chunk_loss.backward()

                trg_loss.update(chunk_trg_loss.item())
                lm_loss.update(chunk_lm_loss.item())

            if self.clip_grad_norm is not None:
                t.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            tqdm_train_dataloader.set_postfix({'trg_loss': trg_loss.get(),
                                               'lm_loss': lm_loss.get()})

        self.writer.add_scalar('train/trg_loss', trg_loss.get(), global_step=self.last_epoch)
        self.writer.add_scalar('train/lm_loss', lm_loss.get(), global_step=self.last_epoch)

    @t.no_grad()
    def _test_epoch(self, test_dataloader):
        tqdm_test_dataloader = tqdm(test_dataloader, desc=f'Test, epoch #{self.last_epoch}')
        self.model.eval()

        all_preds = []
        for less_toxic_tokens, more_toxic_tokens in tqdm_test_dataloader:
            less_toxic_tokens = less_toxic_tokens.to(self.device)
            more_toxic_tokens = more_toxic_tokens.to(self.device)

            less_toxic_preds, _, _ = self.model(less_toxic_tokens)
            more_toxic_preds, _, _ = self.model(more_toxic_tokens)

            preds = (less_toxic_preds < more_toxic_preds).view(-1).tolist()
            all_preds.extend(preds)

            tqdm_test_dataloader.set_postfix({'rng_acc': sum(all_preds, 0) / len(all_preds)})

        self.writer.add_scalar('test/rank_acc', sum(all_preds, 0) / len(all_preds), global_step=self.last_epoch)

        result_metric = sum(all_preds, 0) / len(all_preds)

        return result_metric

    def _save_checkpoint(self, checkpoint_path):  
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        t.save(self.model.state_dict(), checkpoint_path)
    
    def _train_collate_func(self, data):
        tokens, targets = zip(*data)
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.model.padding_idx)
        targets = t.stack(targets, dim=0)
        
        return tokens, targets

    def _test_collate_func(self, data):
        less_toxic_tokens, more_toxic_tokens = zip(*data)
        less_toxic_tokens = pad_sequence(less_toxic_tokens, batch_first=True, padding_value=self.model.padding_idx)
        more_toxic_tokens = pad_sequence(more_toxic_tokens, batch_first=True, padding_value=self.model.padding_idx)

        return less_toxic_tokens, more_toxic_tokens

    def train(self, train_data, n_epochs, batch_size, test_data=None, test_batch_size=None,
              test_each=1, last_checkpoint_path=None, best_checkpoint_path=None):

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                      collate_fn=self._train_collate_func, num_workers=self.n_jobs,
                                      persistent_workers=True)

        if test_data is not None:
            if test_batch_size is None:
                test_batch_size = batch_size
            test_dataloader = DataLoader(test_data, batch_size=test_batch_size, collate_fn=self._test_collate_func,
                                         num_workers=self.n_jobs, persistent_workers=True)

        best_metric = float('-inf')
        for _ in range(n_epochs):
            self._train_epoch(train_dataloader)

            if last_checkpoint_path is not None:
                self._save_checkpoint(last_checkpoint_path)

            if test_data is not None and (self.last_epoch + 1) % test_each == 0:
                metric = self._test_epoch(test_dataloader)
                
                if best_checkpoint_path is not None and metric > best_metric:
                    best_metric = metric
                    self._save_checkpoint(best_checkpoint_path)

            self.last_epoch += 1
