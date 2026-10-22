"""RGB-only dataset for complete official-video inference.

Unlike ActionClipDataset, this dataset does not depend on per-frame label files.
Every image in the supplied official manifest becomes one prediction target.
"""
from pathlib import Path
from bisect import bisect_right

import torch
from PIL import Image
from torch.utils.data import Dataset

from .clip import ClipAugment


class FullRGBVideoDataset(Dataset):
    def __init__(self, root, manifest, model_cfg, train_cfg):
        if model_cfg.keyframe % model_cfg.clip_length != model_cfg.clip_length - 1:
            raise ValueError('this model predicts the last frame; use keyframe=-1')
        self.root = Path(root)
        self.cfg = model_cfg
        self.tc = train_cfg
        self.transform = ClipAugment(
            model_cfg.image_size, train=False, jitter=train_cfg.jitter,
            hue=train_cfg.hue, saturation=train_cfg.saturation,
            exposure=train_cfg.exposure,
        )
        self.frames = []
        self.video_starts = []
        self.video_paths = []
        for video in sorted(manifest):
            paths = list(manifest[video])
            if not paths:
                raise ValueError(f'{video}: RGB manifest is empty')
            ids = [int(Path(path).stem) for path in paths]
            if ids != list(range(1, len(paths) + 1)):
                raise ValueError(f'{video}: expected RGB frames numbered 1..N')
            start = len(self.frames)
            normalized = [Path(path) for path in paths]
            self.frames.extend((video, path, start) for path in normalized)
            self.video_starts.append(start)
            self.video_paths.append(normalized)
        if not self.frames:
            raise ValueError('RGB manifest contains no frames')

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        video, target_path, video_start = self.frames[index]
        end_in_video = index - video_start
        paths = self.video_paths[bisect_right(self.video_starts, index) - 1]
        frames = []
        for offset in reversed(range(self.cfg.clip_length)):
            position = max(0, end_in_video - offset * self.tc.sampling_rate)
            with Image.open(paths[position]) as image:
                frames.append(image.convert('RGB'))
        original_hw = (frames[-1].height, frames[-1].width)
        if any(image.size != frames[-1].size for image in frames):
            raise ValueError(f'{video}: frames in an inference clip have different image sizes')
        boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.long)
        clip, _, _ = self.transform(frames, boxes, labels)
        image_id = f'{video}/{target_path.name}'
        target = {'image_id': image_id, 'original_hw': original_hw}
        return clip, target


def collate_video_frames(batch):
    clips, targets = zip(*batch)
    return torch.stack(clips), list(targets)
