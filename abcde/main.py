import argparse
import os
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
torch.set_num_threads(16)
torch.set_num_interop_threads(1)
from core import load_config, build_model, DetectionCriterion
from cfg.defaults import config_dict
from datasets.list_dataset import ActionClipDataset, collate_clips
from core.optimization import seed_everything, seed_worker, WarmupCosine, WarmupMultiStep, train_one_epoch, collect_detections
from evaluation_ucf24_jhmdb.eval_results import frame_map

def atomic_checkpoint(checkpoint, path):
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', '--config', dest='config', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--train-list', required=True)
    p.add_argument('--test-list', required=True,
                   help='official test frame list; evaluated each epoch for historical model selection')
    p.add_argument('--labels-root', help='annotation directory; defaults to <root>/labels')
    p.add_argument('--output', required=True)
    p.add_argument('--resume')
    p.add_argument('--device', default='cuda')
    p.add_argument('--annotation-version', required=True, help='Record corrected annotation release or official JHMDB split')
    args = p.parse_args()
    mc, lc, tc = load_config(args.config)
    if tc.accumulation_steps < 1 or tc.batch_size < 1 or tc.epochs < 1:
        raise ValueError('invalid training counts')
    seed_everything(tc.seed)
    device = torch.device(args.device)
    model = build_model(mc).to(device)
    criterion = DetectionCriterion(mc.num_classes, lc)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    generator = torch.Generator().manual_seed(tc.seed)
    loaders = []
    for train, list_file in ((True, args.train_list), (False, args.test_list)):
        dataset = ActionClipDataset(args.root, list_file, mc, tc, train,
                                    annotation_version=args.annotation_version,
                                    labels_root=args.labels_root)
        loaders.append(DataLoader(dataset, batch_size=tc.batch_size, shuffle=train, num_workers=tc.workers, collate_fn=collate_clips, pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=generator if train else None, drop_last=False))
    training, testing = loaders
    updates_per_epoch = math.ceil(len(training) / tc.accumulation_steps)
    if tc.lr_schedule == 'multistep':
        scheduler = WarmupMultiStep(optimizer, tc.epochs * updates_per_epoch, tc.warmup_steps,
                                   updates_per_epoch, tc.lr_milestones, tc.lr_gamma)
    else:
        scheduler = WarmupCosine(optimizer, tc.epochs * updates_per_epoch, tc.warmup_steps)
    scaler = (torch.amp.GradScaler('cuda', enabled=tc.amp and device.type == 'cuda')
              if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=tc.amp and device.type == 'cuda'))
    start, best = (0, -1.0)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get('test_evaluation_pending') or checkpoint.get('validation_pending'):
            raise ValueError('This checkpoint precedes official test evaluation; evaluate it before resuming to avoid skipping an epoch evaluation')
        if checkpoint['config'] != config_dict(mc, lc, tc):
            raise ValueError('resume config differs from checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best = (checkpoint['epoch'] + 1, checkpoint['best'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        generator.set_state(checkpoint['loader_rng'].cpu())
        if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'arguments': vars(args), 'config': config_dict(mc, lc, tc),
                'train_frames': len(training.dataset), 'test_frames': len(testing.dataset),
                'model_selection_split': 'official_test',
                'torch_version': torch.__version__, 'device': str(device)}
    (output / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)
    for epoch in range(start, tc.epochs):
        epoch_started = time.monotonic()
        print(json.dumps({'phase': 'epoch_start', 'epoch': epoch + 1, 'epochs': tc.epochs,
                          'lr': optimizer.param_groups[0]['lr'],
                          'time_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
        train_stats = train_one_epoch(model, criterion, training, optimizer, device, tc, scaler, scheduler)
        if isinstance(scheduler, WarmupMultiStep):
            scheduler.step_epoch()
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'config': config_dict(mc, lc, tc), 'annotation_version': args.annotation_version, 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'loader_rng': generator.get_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, 'test_evaluation_pending': True, 'train_stats': train_stats}
        atomic_checkpoint(checkpoint, output / 'pre_test_evaluation.pt')
        print(json.dumps({'phase': 'checkpoint_before_test_evaluation', 'epoch': epoch + 1, 'path': str(output / 'pre_test_evaluation.pt')}), flush=True)
        predictions, truths = collect_detections(model, testing, device, tc)
        evaluation = frame_map(predictions, truths, mc.num_classes)
        score = evaluation['frame_map']
        is_best = score is not None and score > best
        if is_best:
            best = score
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'config': config_dict(mc, lc, tc), 'annotation_version': args.annotation_version, 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'loader_rng': generator.get_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        atomic_checkpoint(checkpoint, output / 'last.pt')
        if is_best:
            atomic_checkpoint(checkpoint, output / 'best.pt')
        record = {'epoch': epoch + 1, 'train': train_stats, 'test_evaluation': evaluation,
                  'model_selection_split': 'official_test',
                  'annotation_version': args.annotation_version}
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        print(json.dumps({'phase': 'epoch_end', 'epoch': epoch + 1,
                          'next_lr': optimizer.param_groups[0]['lr'],
                          'frame_map_05_percent': score * 100 if score is not None else None,
                          'best_percent': best * 100,
                          'elapsed_seconds': round(time.monotonic() - epoch_started, 1)}), flush=True)
    print(json.dumps({'phase': 'training_complete', 'epochs': tc.epochs,
                      'best_percent': best * 100,
                      'time_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
if __name__ == '__main__':
    main()
