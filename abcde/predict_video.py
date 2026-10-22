"""Export detections for every RGB frame in an official test split.

The JSON schema matches evaluate.py's frame_detections.json and is accepted by
evaluate_video.py. Coordinates are in the model's square input image space.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from cfg.defaults import ModelConfig, TrainConfig
from core import build_model
from datasets.full_video import FullRGBVideoDataset, collate_video_frames
from evaluation_ucf24_jhmdb.video_map import frame_manifest, official_videos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', choices=('ucf24', 'jhmdb21'), required=True)
    parser.add_argument('--split', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='new JSON file; existing files are refused')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error('output already exists; choose a new path to preserve existing results')
    if args.batch_size < 1 or args.workers < 0:
        parser.error('--batch-size must be positive and --workers nonnegative')
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_cfg = ModelConfig(**checkpoint['config']['model'])
    model_cfg.pretrained = False
    train_cfg = TrainConfig(**checkpoint['config']['train'])
    videos = official_videos(args.root, args.dataset, args.split)
    manifest = frame_manifest(args.root, videos)
    dataset = FullRGBVideoDataset(args.root, manifest, model_cfg, train_cfg)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_video_frames,
        pin_memory=device.type == 'cuda',
    )
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    if temporary.exists():
        parser.error(f'temporary output already exists: {temporary}; remove it after checking the prior run')
    started = time.monotonic()
    records = 0
    try:
        with torch.inference_mode(), temporary.open('w', encoding='utf-8') as stream:
            stream.write('[')
            for clips, targets in loader:
                predictions = model.predict(
                    clips.to(device, non_blocking=True),
                    train_cfg.score_threshold, train_cfg.nms_threshold,
                )
                for target, prediction in zip(targets, predictions):
                    record = {
                        'image_id': target['image_id'],
                        **{key: value.detach().cpu().tolist()
                           for key, value in prediction.items()},
                    }
                    if records:
                        stream.write(',')
                    stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False))
                    records += 1
                if records % 5000 < len(targets) or records == len(dataset):
                    stream.flush()
                    print(json.dumps({
                        'phase': 'full_rgb_inference', 'dataset': args.dataset,
                        'split': args.split, 'frames': records,
                        'total_frames': len(dataset), 'videos': len(videos),
                        'elapsed_seconds': round(time.monotonic() - started, 1),
                        'gpu_reserved_gb': round(torch.cuda.memory_reserved(device) / 1024**3, 2)
                        if device.type == 'cuda' else None,
                    }), flush=True)
            stream.write(']\n')
            stream.flush()
            os.fsync(stream.fileno())
        if records != len(dataset):
            raise RuntimeError(f'exported {records} records for {len(dataset)} RGB frames')
        # Linking atomically publishes the finished file without clobbering a
        # result that another process may have created after the initial check.
        os.link(temporary, args.output)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({
        'phase': 'prediction_export_complete', 'path': str(args.output),
        'records': records, 'videos': len(videos),
        'coordinate_space': 'input', 'input_size': [model_cfg.image_size] * 2,
        'elapsed_seconds': round(time.monotonic() - started, 1),
    }), flush=True)


if __name__ == '__main__':
    main()
