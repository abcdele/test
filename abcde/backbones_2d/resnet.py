import torch
from torch import nn
from pathlib import Path
import re


def load_video_pretrained(backbone, path):
    """Load MMAction TSN weights by name, checking every backbone tensor."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    source = checkpoint.get('state_dict', checkpoint)
    state = {}
    for key, value in source.items():
        if key.startswith('cls_head.'):
            continue
        key = key.removeprefix('backbone.')
        key = key.replace('conv1.conv.', 'conv1.') if key.startswith('conv1.conv.') else key
        key = key.replace('conv1.bn.', 'bn1.') if key.startswith('conv1.bn.') else key
        key = re.sub(r'\.conv([123])\.conv\.', r'.conv\1.', key)
        key = re.sub(r'\.conv([123])\.bn\.', r'.bn\1.', key)
        key = key.replace('.downsample.conv.', '.downsample.0.')
        key = key.replace('.downsample.bn.', '.downsample.1.')
        if key in state:
            raise ValueError(f'duplicate pretrained tensor: {key}')
        state[key] = value
    # The detector excludes the ImageNet classifier; all 318 backbone tensors
    # must match, including BN statistics. Never silently skip mismatches.
    backbone.fc = nn.Identity()
    backbone.load_state_dict(state, strict=True)

class FrameResNet(nn.Module):

    def __init__(self, name='resnet50', pretrained=True, freeze_bn=False,
                 pretrained_path=None, freeze_early=False):
        super().__init__()
        from torchvision import models
        choices = {'resnet18': (models.resnet18, models.ResNet18_Weights, 512), 'resnet50': (models.resnet50, models.ResNet50_Weights, 2048), 'resnet101': (models.resnet101, models.ResNet101_Weights, 2048)}
        if name not in choices:
            raise ValueError(name)
        factory, weights, self.out_channels = choices[name]
        backbone = factory(weights=weights.DEFAULT if pretrained and not pretrained_path else None)
        if pretrained and pretrained_path:
            load_video_pretrained(backbone, pretrained_path)
        if freeze_early:
            for module in (backbone.conv1, backbone.bn1, backbone.layer1):
                module.requires_grad_(False)
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for layer in self.body.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, clip):
        b, t, c, h, w = clip.shape
        frames = clip.reshape(b * t, c, h, w)
        features = self.body(frames)
        return features.reshape(b, t, *features.shape[1:])
