import math
import torch

def box_iou(a, b, eps=1e-07):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (rb - lt).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_a[:, None] + area_b[None] - intersection + eps)

def aligned_ciou(a, b, eps=1e-07):
    a, b = (a.float(), b.float())
    awh = (a[:, 2:] - a[:, :2]).clamp_min(eps)
    bwh = (b[:, 2:] - b[:, :2]).clamp_min(eps)
    intersection = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(-1)
    iou = intersection / (awh.prod(-1) + bwh.prod(-1) - intersection + eps)
    center_distance = ((a[:, :2] + a[:, 2:] - b[:, :2] - b[:, 2:]) * 0.5).square().sum(-1)
    diagonal = (torch.maximum(a[:, 2:], b[:, 2:]) - torch.minimum(a[:, :2], b[:, :2])).square().sum(-1)
    v = 4 / math.pi ** 2 * (torch.atan(bwh[:, 0] / bwh[:, 1]) - torch.atan(awh[:, 0] / awh[:, 1])).square()
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - center_distance / (diagonal + eps) - alpha * v

def classwise_nms(boxes, scores, labels, threshold=0.5, max_detections=100):
    selected = []
    for label in labels.unique():
        indices = torch.nonzero(labels == label).flatten()
        order = indices[torch.argsort(scores[indices], descending=True, stable=True)]
        while len(order):
            index = order[0]
            selected.append(index)
            if len(order) == 1:
                break
            overlaps = box_iou(boxes[index:index + 1], boxes[order[1:]])[0]
            order = order[1:][overlaps <= threshold]
    if not selected:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    indices = torch.stack(selected)
    ordered = indices[torch.argsort(scores[indices], descending=True, stable=True)]
    return ordered[:max_detections] if max_detections>0 else ordered
