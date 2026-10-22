from __future__ import annotations
from dataclasses import dataclass, asdict
import json

@dataclass
class ModelConfig:
    num_classes: int = 24
    backbone: str = 'resnet50'
    pretrained: bool = True
    backbone_pretrained_path: str | None = 'pretrained/tsn_r50_8xb32_1x1x8_100e_kinetics400_rgb_20220818-2692d16c.pth'
    freeze_backbone_early: bool = False
    image_size: int = 224
    clip_length: int = 16
    num_points: int = 4
    predictor_hidden: int = 64
    gaussian_sigma: float = 0.3
    mask_normalization: str = 'peak'
    motion_alpha: float = 0.3
    mamba_backend: str = 'official'
    d_state: int = 16
    d_conv: int = 4
    expand: float = 2.0
    bidirectional: bool = True
    num_layers: int = 1
    dilations: tuple = (1,)
    projection_hidden: int = 384
    head_dim: int = 256
    head_depth: int = 4
    cls_depth: int | None = None
    reg_depth: int | None = None
    head_norm: str = 'none'
    head_activation: str = 'relu'
    center_offset_clamp: float | None = 32.0
    anchors: tuple = (16.0, 32.0, 64.0, 128.0, 256.0)
    keyframe: int = -1
    sequence_chunk_size: int = 0
    freeze_backbone_bn: bool = False

    def __post_init__(self):
        if self.num_classes < 1 or self.num_points < 1:
            raise ValueError('num_classes and num_points must be positive')
        if self.clip_length < 2 or self.image_size < 32:
            raise ValueError('clip_length >= 2 and image_size >= 32 required')
        if self.num_layers != len(self.dilations) or min(self.dilations) < 1:
            raise ValueError('one positive dilation per temporal layer required')
        if not 0 <= self.motion_alpha <= 1 or self.gaussian_sigma <= 0:
            raise ValueError('invalid motion coefficient or Gaussian bandwidth')
        if self.mask_normalization not in ('minmax', 'peak'):
            raise ValueError('mask normalization must be minmax or peak')
        if self.head_norm not in ('none', 'batch', 'group'):
            raise ValueError('invalid head_norm')
        if self.head_activation not in ('relu', 'silu', 'gelu'):
            raise ValueError('invalid head_activation')
        if self.d_state<1 or self.d_conv<1 or self.expand<=0:
            raise ValueError('invalid temporal dimensions')
        if self.center_offset_clamp is not None and self.center_offset_clamp<=0:
            raise ValueError('center_offset_clamp must be positive or null')
        if not self.anchors or any(v<=0 for a in self.anchors for v in (a if isinstance(a,(tuple,list)) else (a,))):
            raise ValueError('anchors must be positive pixel sizes')
        if not -self.clip_length <= self.keyframe < self.clip_length:
            raise ValueError('keyframe outside clip')

@dataclass
class LossConfig:
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    cls_weight: float = 1.0
    box_weight: float = 1.0
    consistency_weight: float = 0.5
    match_topk: int = 4
    positive_iou: float = 0.15
    ignore_iou: float = 0.7
    targets_normalized: bool = False

@dataclass
class TrainConfig:
    epochs: int = 7
    batch_size: int = 6
    accumulation_steps: int = 18
    lr: float = 0.0001
    weight_decay: float = 0.0001
    warmup_steps: int = 50
    lr_schedule: str = 'cosine'
    lr_milestones: tuple = (1, 2, 3, 4)
    lr_gamma: float = 0.5
    amp: bool = True
    grad_clip: float = 5.0
    seed: int = 42
    workers: int = 4
    sampling_rate: int = 1
    jitter: float = 0.2
    hue: float = 0.1
    saturation: float = 1.5
    exposure: float = 1.5
    score_threshold: float = 0.005
    nms_threshold: float = 0.5

    def __post_init__(self):
        if self.lr_schedule not in ('cosine', 'multistep'):
            raise ValueError('invalid learning rate schedule')
        if not 0 < self.lr_gamma <= 1:
            raise ValueError('lr_gamma must be in (0, 1]')
        if any(not isinstance(e, int) or e < 1 for e in self.lr_milestones):
            raise ValueError('learning rate milestones must be positive epochs')

def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    m = dict(cfg.get('model', {}))
    for k in ('anchors', 'dilations'):
        if k in m and isinstance(m[k],list):
            m[k] = tuple(tuple(a) if isinstance(a,list) else a for a in m[k])
    return (ModelConfig(**m), LossConfig(**cfg.get('loss', {})), TrainConfig(**cfg.get('train', {})))

def config_dict(model, loss, train):
    return {'model': asdict(model), 'loss': asdict(loss), 'train': asdict(train)}
