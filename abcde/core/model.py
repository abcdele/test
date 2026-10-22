from backbones_2d.resnet import FrameResNet
import torch
from torch import nn
from cfg.defaults import ModelConfig
from .alignment import EnhancedILA, ConvBNAct
from .temporal import DilatedSpatialTemporalMamba
from .head import DecoupledAnchorHead

class MIAActionDetector(nn.Module):

    def __init__(self, cfg=None, backbone=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.backbone = backbone if backbone is not None else FrameResNet(
            self.cfg.backbone, self.cfg.pretrained, self.cfg.freeze_backbone_bn,
            self.cfg.backbone_pretrained_path, self.cfg.freeze_backbone_early)
        c = self.backbone.out_channels
        self.alignment = EnhancedILA(c, self.cfg.num_points, self.cfg.predictor_hidden, self.cfg.gaussian_sigma, self.cfg.motion_alpha, self.cfg.mask_normalization)
        self.temporal = DilatedSpatialTemporalMamba(c, self.cfg)
        self.projection = nn.Sequential(ConvBNAct(c, self.cfg.projection_hidden, 3, 'relu'), ConvBNAct(self.cfg.projection_hidden, self.cfg.head_dim, 1, 'relu'))
        self.head = DecoupledAnchorHead(self.cfg.head_dim, self.cfg.num_classes, self.cfg.anchors, self.cfg.head_depth, self.cfg.cls_depth, self.cfg.reg_depth, self.cfg.head_norm, self.cfg.head_activation, self.cfg.center_offset_clamp)

    def forward(self, clip, return_aux=False):
        if clip.ndim != 5 or clip.shape[1] != self.cfg.clip_length or clip.shape[2] != 3:
            raise ValueError(f'expected B,{self.cfg.clip_length},3,H,W input')
        features = self.backbone(clip)
        tokens, aux = (None, {})
        features, tokens, aux = self.alignment(features)
        modeled = self.temporal(features, tokens)
        key_features = self.projection(modeled[:, self.cfg.keyframe])
        output = self.head(key_features, clip.shape[-2:])
        output['interaction_tokens'] = tokens
        if return_aux:
            output['aux'] = {**aux, 'modeled_features': modeled, 'keyframe_features': key_features}
        return output

    @torch.no_grad()
    def predict(self, clip, score_threshold=0.005, nms_threshold=0.5, max_detections=0, normalized=False):
        if self.training:
            raise RuntimeError('call model.eval() before predict')
        return self.head.postprocess(self(clip), score_threshold, nms_threshold, max_detections, normalized=normalized)

def build_model(cfg):
    return MIAActionDetector(cfg)
