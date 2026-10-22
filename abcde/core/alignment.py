import torch
from torch import nn

class ConvBNAct(nn.Sequential):

    def __init__(self, cin, cout, kernel=3, activation='silu'):
        act = nn.SiLU(inplace=True) if activation == 'silu' else nn.ReLU(inplace=True)
        super().__init__(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False), nn.BatchNorm2d(cout), act)

class MultiPointPredictor(nn.Module):

    def __init__(self, channels, num_points=4, hidden=64):
        super().__init__()
        self.num_points = num_points
        reduced = max(8, hidden // 2)
        self.net = nn.Sequential(ConvBNAct(2 * channels, hidden), nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(hidden, reduced, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(reduced, 3 * num_points, 1))
        nn.init.normal_(self.net[-1].weight, std=0.001)
        nn.init.zeros_(self.net[-1].bias)
        side = int(num_points ** 0.5 + 0.999)
        axis = torch.linspace(-0.6, 0.6, side) if side > 1 else torch.zeros(1)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        coords = torch.stack((xx.flatten(), yy.flatten()), -1)[:num_points]
        with torch.no_grad():
            self.net[-1].bias.view(num_points, 3)[:, :2].copy_(torch.atanh(coords))

    def forward(self, pair):
        prediction = self.net(pair).flatten(1).reshape(-1, self.num_points, 3)
        return (prediction[..., :2].tanh(), prediction[..., 2].softmax(-1))

class GaussianMixtureMask(nn.Module):

    def __init__(self, sigma=0.3, normalization='peak'):
        super().__init__()
        if sigma <= 0 or normalization not in ('minmax', 'peak'):
            raise ValueError('invalid Gaussian mask settings')
        self.sigma, self.normalization = (sigma, normalization)

    def forward(self, points, weights, height, width):
        y = torch.linspace(-1, 1, height, device=points.device, dtype=torch.float32)
        x = torch.linspace(-1, 1, width, device=points.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist2 = (xx[None, None] - points.float()[..., 0, None, None]).square() + (yy[None, None] - points.float()[..., 1, None, None]).square()
        raw = (weights.float()[..., None, None] * torch.exp(-dist2 / (2 * self.sigma ** 2))).sum(1)
        maximum = raw.amax((-2, -1), keepdim=True)
        if self.normalization == 'minmax':
            minimum = raw.amin((-2, -1), keepdim=True)
            span = maximum - minimum
            scaled = (raw - minimum) / span.clamp_min(1e-06)
            scaled = torch.where(span > 1e-06, scaled, torch.full_like(scaled, 0.5))
        else:
            scaled = raw / (maximum + 1e-6)
        return (0.5 + scaled).unsqueeze(1)

class MotionIntensityEstimator(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(2 * channels, 32), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, pair):
        return self.net(pair).view(-1, 1, 1, 1)

class EnhancedILA(nn.Module):

    def __init__(self, channels, num_points=4, hidden=64, sigma=0.3, alpha=0.3, normalization='peak'):
        super().__init__()
        self.channels, self.alpha = channels, alpha
        self.point_predictor = MultiPointPredictor(channels, num_points, hidden)
        self.mask_generator = GaussianMixtureMask(sigma, normalization)
        self.motion_estimator = MotionIntensityEstimator(channels)
        self.token_projection = nn.Sequential(nn.Linear(channels, channels), nn.LayerNorm(channels), nn.SiLU())

    def _process_pairs(self, previous, current):
        _, c, h, w = current.shape
        pair = torch.cat((current, previous), dim=1)
        points, weights = self.point_predictor(pair)
        mask = self.mask_generator(points, weights, h, w).to(current.dtype)
        aligned = current * mask
        score = self.motion_estimator(pair)
        coefficient = self.alpha * score
        fused = current + coefficient * (aligned-current)
        # MIA's interaction token is pooled after applying the Gaussian mask twice.
        source = aligned * mask
        token_source = source.mean((-2, -1))
        token = self.token_projection(token_source)
        return fused, token, points, weights, mask, score

    def forward(self, features):
        b,t,c,h,w = features.shape
        if t<2 or c!=self.channels:
            raise ValueError('expected at least two frames with matching channels')
        # Keep the same pair construction in train and eval.  The previous
        # condition only selected framewise processing while training, so an
        # eval pass silently flattened all (T-1) pairs into one large batch.
        # That changes BatchNorm statistics in the point/motion predictors and
        # makes validation use a different feature distribution than training.
        results = [self._process_pairs(features[:,i-1],features[:,i]) for i in range(1,t)]
        values = [torch.stack(items,1) for items in zip(*results)]
        fused,tokens,points,weights,mask,score = values
        sequence = torch.cat((features[:,:1],fused),1)
        return sequence,tokens,{'points':points,'point_weights':weights,'masks':mask,'motion_scores':score.reshape(b,t-1)}
