import math
import torch
from torch import nn
from torch.nn import functional as F

class ReferenceMamba(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2.0):
        super().__init__()
        self.inner = int(expand * d_model)
        self.state_dim, self.dt_rank = (d_state, math.ceil(d_model / 16))
        self.in_proj = nn.Linear(d_model, 2 * self.inner, bias=False)
        self.conv1d = nn.Conv1d(self.inner, self.inner, d_conv, groups=self.inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner)
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** (-0.5), self.dt_rank ** (-0.5))
        dt = torch.exp(torch.rand(self.inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))
        delta, B, C = torch.split(self.x_proj(u), [self.dt_rank, self.state_dim, self.state_dim], -1)
        delta = F.softplus(self.dt_proj(delta)).float()
        A = -self.A_log.float().exp()
        state = torch.zeros(x.shape[0], self.inner, self.state_dim, device=x.device, dtype=torch.float32)
        values = []
        for t in range(x.shape[1]):
            step = delta[:, t, :, None]
            state = (step * A).exp() * state + step * B[:, t, None, :].float() * u[:, t, :, None].float()
            values.append((state * C[:, t, None, :].float()).sum(-1) + self.D.float() * u[:, t].float())
        out = torch.stack(values, 1).to(u.dtype) * F.silu(z)
        return self.out_proj(out)

def make_mamba(channels, cfg):
    if cfg.mamba_backend == 'reference':
        return ReferenceMamba(channels, cfg.d_state, cfg.d_conv, cfg.expand)
    if cfg.mamba_backend != 'official':
        raise ValueError('backend must be official or reference')
    try:
        from mamba_ssm import Mamba
    except ImportError as e:
        raise ImportError('Install mamba-ssm for CUDA training; use mamba_backend=reference explicitly for CPU verification.') from e
    return Mamba(d_model=channels, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)

class MITokenFusionGate(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Linear(2 * channels, channels)
        self.value = nn.Linear(channels, channels)

    def forward(self, spatial_sequences, frame_tokens):
        token = frame_tokens[:, None].expand(-1, spatial_sequences.shape[1], -1, -1)
        gate = torch.sigmoid(self.gate(torch.cat((spatial_sequences, token), -1)))
        delta = gate * self.value(token)
        return spatial_sequences + delta

class BidirectionalSSMBlock(nn.Module):

    def __init__(self, channels, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.forward_ssm = make_mamba(channels, cfg)
        self.backward_ssm = make_mamba(channels, cfg) if cfg.bidirectional else None
        self.chunk_size = cfg.sequence_chunk_size
        self.direction_projection = nn.Linear(2*channels,channels) if cfg.bidirectional else None

    def _scan(self, x):
        y = self.forward_ssm(x)
        if self.backward_ssm is not None:
            reverse = self.backward_ssm(x.flip(1)).flip(1)
            y = self.direction_projection(torch.cat((y,reverse),-1))
        return y

    def forward(self, x, dilation=1):
        normalized = self.norm(x)
        output = torch.zeros_like(x)
        for offset in range(min(dilation, x.shape[1])):
            sub = normalized[:, offset::dilation]
            if self.chunk_size > 0:
                y = torch.cat([self._scan(z) for z in sub.split(self.chunk_size, 0)], 0)
            else:
                y = self._scan(sub)
            output[:, offset::dilation] = y
        return output

class DilatedSpatialTemporalMamba(nn.Module):

    def __init__(self, channels, cfg):
        super().__init__()
        self.cfg, self.channels = (cfg, channels)
        self.layers = nn.ModuleList([BidirectionalSSMBlock(channels, cfg) for _ in range(cfg.num_layers)])
        self.gates = nn.ModuleList([MITokenFusionGate(channels) for _ in range(cfg.num_layers)])
    def forward(self, features, tokens=None):
        b, t, c, h, w = features.shape
        sequences = features.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        if tokens is None or tokens.shape != (b, t - 1, c):
            raise ValueError('expected B,T-1,C interaction tokens')
        frame_tokens = torch.cat((tokens[:, :1], tokens), 1)
        for i, (layer, dilation) in enumerate(zip(self.layers, self.cfg.dilations)):
            residual = sequences
            update = layer(sequences, dilation)
            update = self.gates[i](update.reshape(b,h*w,t,c),frame_tokens).reshape(b*h*w,t,c)
            sequences = residual + update
        return sequences.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2).contiguous()
