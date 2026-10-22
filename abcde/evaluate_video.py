"""CPU-only Video-mAP from saved frame predictions; no model weights required."""
import argparse
import json
from pathlib import Path
from evaluation_ucf24_jhmdb.video_map import (
    THRESHOLDS, CLASSES, official_videos, frame_manifest, ground_truth,
    predictions_to_tubes, video_map,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=THRESHOLDS, required=True)
    p.add_argument('--split', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True, help='evaluate.py frame_detections.json (0-based labels)')
    p.add_argument('--coordinate-space', choices=('original', 'input'), required=True)
    p.add_argument('--input-size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'))
    p.add_argument('--annotations', type=Path, help='UCF24 finalAnnots.mat override')
    p.add_argument('--output', type=Path, required=True, help='NEW output directory; existing paths are refused')
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('output already exists; choose a new directory to preserve existing results')
    videos = official_videos(args.root, args.dataset, args.split)
    manifest = frame_manifest(args.root, videos)
    truths = ground_truth(args.root, args.dataset, manifest, args.annotations)
    records = json.loads(args.predictions.read_text())
    if not isinstance(records, list):
        p.error('predictions must be a JSON list of frame records')
    tubes = predictions_to_tubes(records, manifest, args.dataset, args.coordinate_space, args.input_size)
    result = {'dataset': args.dataset, 'split': args.split, 'classes': CLASSES[args.dataset],
              'protocol': 'YOWO Viterbi top3 / tube NMS 0.3 / mean spatial IoU * temporal IoU / integral VOC AP',
              'reference_commit': '9d202c1694dfa883e67fd8c5455a2b7ad779684f',
              'coordinate_space': args.coordinate_space, 'input_size': args.input_size,
              'num_videos': len(videos), 'num_frames': sum(map(len, manifest.values())),
              'num_predicted_tubes': len(tubes), 'num_gt_tubes': len(truths),
              'metrics': video_map(tubes, truths, len(CLASSES[args.dataset]), THRESHOLDS[args.dataset])}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'video_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output/'predicted_tubes.json').write_text(json.dumps(
        [{**t, 'boxes': t['boxes'].tolist()} for t in tubes], allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
