import os
import random

import torch as t
import torch.nn.functional as F
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


    def __init__(self, model, optimizer_params={}, loss_params={}, clip_grad_norm=None,
                 tb_dir=None, device=None, n_jobs=1, seed=0):
        self._set_seed(seed)

        if device is None:
            device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        else:
            device = t.device(device)

        self.model = model.to(device)

        trg_weight = loss_params.get('trg_weight', 1)
        lm_weight = loss_params.get('lm_weight', 0)
        smoothing = loss_params.get('smoothing', 0)
        self.trg_criterion = lambda *args: trg_weight * F.mse_loss(*args)
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

        self.device = device
        self.n_jobs = n_jobs

        self.last_epoch = 0

    def _train_epoch(self, train_dataloader):
        tqdm_train_dataloader = tqdm(train_dataloader, desc=f'Train, epoch #{self.last_epoch}')
        self.model.train()

        trg_loss, lm_loss = AvgValue(), AvgValue()
        for tokens, targets in tqdm_train_dataloader:
            tokens = tokens.to(self.device)
            targets = targets.to(self.device)
            preds, lm_logits, padding_mask = self.model(tokens)

            batch_trg_loss = self.trg_criterion(preds, targets)
            batch_lm_loss = self.lm_criterian(lm_logits[:, :-1], tokens[:, 1:], padding_mask[:, 1:])

            self.optimizer.zero_grad()
            (batch_trg_loss + batch_lm_loss).backward()
            if self.clip_grad_norm is not None:
                t.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            trg_loss.update(batch_trg_loss.item())
            lm_loss.update(batch_lm_loss.item())

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

        self.writer.add_scalar('test/rng_acc', sum(all_preds, 0) / len(all_preds), global_step=self.last_epoch)

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
