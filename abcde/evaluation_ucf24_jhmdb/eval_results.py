import json
from pathlib import Path
import torch
from core.utils import box_iou

def frame_map(predictions, truths, num_classes, iou_threshold=0.5):
    if set(predictions) != set(truths):
        raise ValueError('prediction and ground-truth frame sets differ')
    ap_by_class, n_gt_by_class = ({}, {})
    for label in range(num_classes):
        gt = {image: target['boxes'][target['labels'] == label] for image, target in truths.items()}
        total = sum((len(boxes) for boxes in gt.values()))
        n_gt_by_class[str(label)] = total
        if total == 0:
            ap_by_class[str(label)] = None
            continue
        candidates = []
        for image, pred in predictions.items():
            for index in torch.nonzero(pred['labels'] == label).flatten().tolist():
                candidates.append((float(pred['scores'][index]), image, pred['boxes'][index]))
        candidates.sort(key=lambda item: item[0], reverse=True)
        assigned = {image: torch.zeros(len(boxes), dtype=torch.bool) for image, boxes in gt.items()}
        tp, fp = ([], [])
        for score, image, box in candidates:
            if len(gt[image]):
                overlap = box_iou(box[None], gt[image])[0]
                value, index = overlap.max(0)
                success = bool(value >= iou_threshold) and (not bool(assigned[image][index]))
                if success:
                    assigned[image][index] = True
            else:
                success = False
            tp.append(float(success))
            fp.append(float(not success))
        if not tp:
            ap_by_class[str(label)] = 0.0
            continue
        tp, fp = (torch.tensor(tp).cumsum(0), torch.tensor(fp).cumsum(0))
        recall, precision = (tp / total, tp / (tp + fp).clamp_min(1e-08))
        recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
        precision = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))
        precision = precision.flip(0).cummax(0).values.flip(0)
        indices = torch.nonzero(recall[1:] != recall[:-1]).flatten()
        ap_by_class[str(label)] = float(((recall[indices + 1] - recall[indices]) * precision[indices + 1]).sum())
    values = [v for v in ap_by_class.values() if v is not None]
    return {'frame_map': sum(values) / len(values) if values else None, 'iou_threshold': iou_threshold, 'ap_by_class': ap_by_class, 'n_gt_by_class': n_gt_by_class, 'evaluated_classes': len(values)}

def export_voc_files(predictions, truths, output_dir, num_classes):
    base = Path(output_dir)
    (base / 'groundtruths').mkdir(parents=True, exist_ok=True)
    (base / 'detections').mkdir(parents=True, exist_ok=True)
    manifest = {}
    for ordinal, image in enumerate(sorted(truths)):
        name = f'{ordinal:08d}.txt'
        manifest[name] = image
        gt, pred = (truths[image], predictions[image])
        gt_lines = [f'{int(label)} ' + ' '.join((f'{float(v):.6f}' for v in box)) for box, label in zip(gt['boxes'], gt['labels'])]
        pred_lines = [f'{int(label)} {float(score):.8f} ' + ' '.join((f'{float(v):.6f}' for v in box)) for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores'])]
        (base / 'groundtruths' / name).write_text('\n'.join(gt_lines))
        (base / 'detections' / name).write_text('\n'.join(pred_lines))
    (base / 'frame_manifest.json').write_text(json.dumps(manifest, indent=2))
