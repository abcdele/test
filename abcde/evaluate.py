import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Set both Torch CPU pools before model/DataLoader construction. Environment
# variables only constrain intra-op/BLAS threads; inter-op otherwise defaults
# to the host CPU count in this PyTorch environment.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from core import build_model
from cfg.defaults import ModelConfig, TrainConfig
from datasets.list_dataset import ActionClipDataset, collate_clips
from core.optimization import collect_detections, seed_worker
from evaluation_ucf24_jhmdb.eval_results import frame_map, export_voc_files

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--labels-root', help='annotation directory; defaults to <root>/labels')
    p.add_argument('--list', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--annotation-version', required=True)
    p.add_argument('--export-voc', action='store_true')
    args = p.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    mc = ModelConfig(**ckpt['config']['model'])
    mc.pretrained = False
    tc = TrainConfig(**ckpt['config']['train'])
    model = build_model(mc).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    dataset = ActionClipDataset(args.root, args.list, mc, tc, False,
                                annotation_version=args.annotation_version,
                                labels_root=args.labels_root)
    loader = DataLoader(dataset, batch_size=tc.batch_size, num_workers=tc.workers, collate_fn=collate_clips, pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    predictions, truths = collect_detections(model, loader, device, tc)
    result = frame_map(predictions, truths, mc.num_classes)
    result['annotation_version'] = args.annotation_version
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    records = []
    for image, pred in predictions.items():
        records.append({'image_id': image, **{k: v.tolist() for k, v in pred.items()}})
    (output / 'frame_detections.json').write_text(json.dumps(records))
    if args.export_voc:
        export_voc_files(predictions, truths, output / 'voc', mc.num_classes)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
