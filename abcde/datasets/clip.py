import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

class ClipAugment:

    def __init__(self, size=224, train=False, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.size, self.train = (size, train)
        self.jitter, self.hue, self.saturation, self.exposure = (jitter, hue, saturation, exposure)

    @staticmethod
    def random_scale(maximum):
        value = random.uniform(1, maximum)
        return value if random.random() < 0.5 else 1 / value

    def __call__(self, frames, boxes, labels):
        width, height = frames[-1].size
        boxes = boxes.clone()
        if self.train:
            dx, dy = (int(width * self.jitter), int(height * self.jitter))
            left, right = (random.randint(-dx, dx), random.randint(-dx, dx))
            top, bottom = (random.randint(-dy, dy), random.randint(-dy, dy))
            crop_w, crop_h = (width - left - right, height - top - bottom)
            if crop_w <= 0 or crop_h <= 0:
                raise ValueError('jitter generated an invalid crop')
            frames = [image.crop((left, top, left + crop_w, top + crop_h)) for image in frames]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            width, height = (crop_w, crop_h)
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
            if random.random() < 0.5:
                frames = [ImageOps.mirror(image) for image in frames]
                old_x1 = boxes[:, 0].clone()
                boxes[:, 0], boxes[:, 2] = (width - boxes[:, 2], width - old_x1)
            hue = random.uniform(-self.hue, self.hue)
            saturation = self.random_scale(self.saturation)
            exposure = self.random_scale(self.exposure)
            changed = []
            for image in frames:
                image = ImageEnhance.Color(image).enhance(saturation)
                image = ImageEnhance.Brightness(image).enhance(exposure)
                hsv = np.array(image.convert('HSV'))
                hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + round(255 * hue)) % 256).astype(np.uint8)
                changed.append(Image.fromarray(hsv, mode='HSV').convert('RGB'))
            frames = changed
        boxes[:, 0::2] *= self.size / width
        boxes[:, 1::2] *= self.size / height
        valid = (boxes[:, 2:] - boxes[:, :2] > 0.001).all(-1)
        boxes, labels = (boxes[valid], labels[valid])
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor_frames = []
        for image in frames:
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255
            tensor_frames.append((tensor - mean) / std)
        return (torch.stack(tensor_frames, 0), boxes, labels)
