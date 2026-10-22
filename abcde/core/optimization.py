import random
import math
import json
import time
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    # Each DataLoader worker is a separate process. Limit its Torch CPU pool;
    # otherwise four workers each create their own pool and validation
    # oversubscribes the host while PIL/feature batches are being prepared.
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)

def move_targets(targets, device):
    return [{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

class WarmupCosine:

    def __init__(self, optimizer, total_updates, warmup_updates):
        self.optimizer = optimizer
        self.total = max(1, total_updates)
        self.warmup = min(warmup_updates, max(0, self.total - 1))
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.index = 0
        self._apply()

    def _apply(self):
        if self.index < self.warmup:
            factor = (self.index + 1) / max(self.warmup, 1)
        else:
            progress = (self.index - self.warmup) / max(1, self.total - self.warmup - 1)
            factor = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
        for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = lr * factor

    def step(self):
        self.index += 1
        self._apply()

    def state_dict(self):
        return {'index': self.index, 'base_lrs': self.base_lrs, 'total': self.total, 'warmup': self.warmup}

    def load_state_dict(self, state):
        self.index, self.base_lrs, self.total, self.warmup = (state['index'], state['base_lrs'], state['total'], state['warmup'])
        self._apply()


class WarmupMultiStep(WarmupCosine):
    """Keep update-counted warmup; decay at completed epoch boundaries."""

    def __init__(self, optimizer, total_updates, warmup_updates, updates_per_epoch,
                 milestones=(1, 2, 3, 4), gamma=0.5):
        self.updates_per_epoch = updates_per_epoch
        self.milestones = tuple(milestones)
        self.gamma = gamma
        self.completed_epochs = 0
        super().__init__(optimizer, total_updates, warmup_updates)

    def _apply(self):
        if self.index < self.warmup:
            factor = (self.index + 1) / max(self.warmup, 1)
        else:
            factor = 1.0
        factor *= self.gamma ** sum(e <= self.completed_epochs for e in self.milestones)
        for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = lr * factor

    def state_dict(self):
        return {**super().state_dict(), 'updates_per_epoch': self.updates_per_epoch,
                'milestones': self.milestones, 'gamma': self.gamma,
                'completed_epochs': self.completed_epochs}

    def step_epoch(self):
        self.completed_epochs += 1
        self._apply()

    def load_state_dict(self, state):
        self.updates_per_epoch = state['updates_per_epoch']
        self.milestones = tuple(state['milestones'])
        self.gamma = state['gamma']
        self.completed_epochs = state['completed_epochs']
        super().load_state_dict(state)

def train_one_epoch(model, criterion, loader, optimizer, device, cfg, scaler, scheduler=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals, count = ({}, 0)
    amp = cfg.amp and device.type == 'cuda'
    if len(loader) == 0:
        raise ValueError('empty loader')
    started = time.monotonic()
    for iteration, (clips, targets) in enumerate(loader):
        window_start = iteration // cfg.accumulation_steps * cfg.accumulation_steps
        window_size = min(cfg.accumulation_steps, len(loader) - window_start)
        clips, targets = (clips.to(device, non_blocking=True), move_targets(targets, device))
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            output = model(clips)
            losses = criterion(output, targets)
            scaled_loss = losses['total'] / window_size
        if not torch.isfinite(scaled_loss):
            raise FloatingPointError(f'non-finite loss at iteration {iteration}')
        scaler.scale(scaled_loss).backward()
        if iteration - window_start + 1 == window_size:
            scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= previous_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        count += len(clips)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0) + float(value.detach()) * len(clips)
        if iteration == 0 or (iteration + 1) % 50 == 0 or iteration + 1 == len(loader):
            print(json.dumps({'phase': 'train', 'batch': iteration + 1, 'batches': len(loader), 'loss': float(losses['total'].detach()), 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    return {name: value / count for name, value in totals.items()}

@torch.no_grad()
def collect_detections(model, loader, device, cfg):
    model.eval()
    records, truths = ({}, {})
    started = time.monotonic()
    for iteration, (clips, targets) in enumerate(loader):
        predictions = model.predict(clips.to(device), cfg.score_threshold, cfg.nms_threshold)
        for target, prediction in zip(targets, predictions):
            image_id = target['image_id']
            if image_id in records:
                raise ValueError('duplicate evaluation image id')
            records[image_id] = {k: v.detach().cpu().clone() for k, v in prediction.items()}
            truths[image_id] = {'boxes': target['boxes'].detach().cpu().clone(), 'labels': target['labels'].detach().cpu().clone()}
        if iteration == 0 or (iteration + 1) % 50 == 0 or iteration + 1 == len(loader):
            print(json.dumps({'phase': 'evaluation', 'batch': iteration + 1,
                              'batches': len(loader), 'frames': len(records),
                              'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    return (records, truths)
