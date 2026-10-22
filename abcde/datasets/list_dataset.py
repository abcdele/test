from .clip import ClipAugment
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps

class ActionClipDataset(Dataset):

    def __init__(self, root, list_file, model_cfg, train_cfg, train=False, class_offset=1, missing_label_is_background=False, annotation_version='unspecified', labels_root=None):
        if model_cfg.keyframe % model_cfg.clip_length != model_cfg.clip_length - 1:
            raise ValueError('this dataset labels the last frame; use keyframe=-1')
        self.root = Path(root)
        self.labels_root = Path(labels_root) if labels_root is not None else self.root / 'labels'
        self.lines = [line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]
        if not self.lines or len(self.lines) != len(set(self.lines)):
            raise ValueError('frame list must be nonempty without duplicate entries')
        self.cfg, self.tc, self.class_offset = (model_cfg, train_cfg, class_offset)
        self.missing_label_is_background = missing_label_is_background
        self.annotation_version = annotation_version
        self.transform = ClipAugment(model_cfg.image_size, train, train_cfg.jitter, train_cfg.hue, train_cfg.saturation, train_cfg.exposure)
        self.frame_cache = {}

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        relative = Path(self.lines[index])
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 3:
            raise ValueError('expected relative class/video/frame path')
        folder = self.root / 'rgb-images' / relative.parent
        if folder not in self.frame_cache:
            paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')), key=lambda p: int(p.stem))
            self.frame_cache[folder] = (paths, {p.name: i for i, p in enumerate(paths)})
        paths, positions = self.frame_cache[folder]
        if relative.name not in positions:
            raise FileNotFoundError(folder / relative.name)
        end = positions[relative.name]
        frames = []
        for t in reversed(range(self.cfg.clip_length)):
            position = max(0, end - t * self.tc.sampling_rate)
            with Image.open(paths[position]) as image:
                frames.append(image.convert('RGB'))
        original_hw = (frames[-1].height, frames[-1].width)
        if any((image.size != frames[-1].size for image in frames)):
            raise ValueError('all frames in a clip must have the same size')
        label_path = self.labels_root / relative.with_suffix('.txt')
        if not label_path.exists():
            if not self.missing_label_is_background:
                raise FileNotFoundError(f'Missing annotation: {label_path}')
            labels_data = np.empty((0, 5), dtype=np.float32)
        elif not label_path.read_text().strip():
            labels_data = np.empty((0, 5), dtype=np.float32)
        else:
            labels_data = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
        if labels_data.shape[1] != 5 or not np.isfinite(labels_data).all():
            raise ValueError('labels must contain five finite columns')
        if len(labels_data) and (not np.equal(labels_data[:, 0], np.floor(labels_data[:, 0])).all()):
            raise ValueError('class ids must be integers')
        boxes = torch.from_numpy(labels_data[:, 1:].copy())
        labels = torch.from_numpy(labels_data[:, 0].copy()).long() - self.class_offset
        if len(labels) and ((labels < 0).any() or (labels >= self.cfg.num_classes).any()):
            raise ValueError('annotation class ids outside configured class range')
        if len(boxes) and (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError('annotation has non-positive box dimensions')
        clip, boxes, labels = self.transform(frames, boxes, labels)
        target = {'boxes': boxes, 'labels': labels, 'image_id': relative.as_posix(), 'original_hw': original_hw, 'annotation_version': self.annotation_version}
        return (clip, target)

def collate_clips(batch):
    clips, targets = zip(*batch)
    return (torch.stack(clips), list(targets))
