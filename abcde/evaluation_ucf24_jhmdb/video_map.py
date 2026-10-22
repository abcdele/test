"""Video-mAP adapters for the YOWO UCF24/J-HMDB evaluation protocol.

Internal labels are zero-based; tubes contain [frame, x1, y1, x2, y2].
Frames are original one-based consecutive video frame IDs, boxes original pixels.
"""
from pathlib import Path, PurePosixPath
import numpy as np
from .yowo_linking import link_bbxes_between_frames, voc_ap

THRESHOLDS = {'ucf24': (0.1, 0.2, 0.5, 0.75), 'jhmdb21': (0.2, 0.5, 0.75)}
CLASSES = {
    'ucf24': 'Basketball BasketballDunk Biking CliffDiving CricketBowling Diving Fencing FloorGymnastics GolfSwing HorseRiding IceDancing LongJump PoleVault RopeClimbing SalsaSpin SkateBoarding Skiing Skijet SoccerJuggling Surfing TennisSwing TrampolineJumping VolleyballSpiking WalkingWithDog'.split(),
    'jhmdb21': 'brush_hair catch clap climb_stairs golf jump kick_ball pick pour pullup push run shoot_ball shoot_bow shoot_gun sit stand swing_baseball throw walk wave'.split(),
}


def tube_array(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 5 or not len(a) or not np.isfinite(a).all():
        raise ValueError('tube must be a nonempty finite Nx5 array')
    if (a[:, 0] < 1).any() or (a[:, 0] != np.floor(a[:, 0])).any() or (np.diff(a[:, 0]) != 1).any():
        raise ValueError('tube frames must be positive, consecutive and increasing')
    if (a[:, 3:5] <= a[:, 1:3]).any():
        raise ValueError('tube boxes must have positive width and height')
    return a


def tube_iou(first, second):
    """Mean spatial IoU on intersection times temporal IoU (YOWO iou3dt).

    Spatial areas use the reference inclusive-pixel +1 convention. Temporal
    endpoints are inclusive too; unlike the old code, a one-frame overlap counts.
    This is NOT the ratio of summed box intersection/union volumes.
    """
    a, b = tube_array(first), tube_array(second)
    start, end = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
    if end < start:
        return 0.0
    aa = a[int(start-a[0, 0]):int(end-a[0, 0])+1, 1:]
    bb = b[int(start-b[0, 0]):int(end-b[0, 0])+1, 1:]
    inter = np.maximum(0, np.minimum(aa[:, 2:], bb[:, 2:]) - np.maximum(aa[:, :2], bb[:, :2]) + 1).prod(1)
    areas = (aa[:, 2:]-aa[:, :2]+1).prod(1) + (bb[:, 2:]-bb[:, :2]+1).prod(1)
    temporal = (end-start+1)/(max(a[-1, 0], b[-1, 0])-min(a[0, 0], b[0, 0])+1)
    return float(np.mean(inter/(areas-inter)) * temporal)


def link_video(frames):
    """Map actual frame IDs to Nx5 [xyxy, score], including empty frames."""
    ids = sorted(frames)
    if not ids or ids[0] < 1 or any(b-a != 1 for a, b in zip(ids, ids[1:])):
        raise ValueError('linking requires a complete consecutive frame sequence')
    linked = link_bbxes_between_frames([np.asarray(frames[i], dtype=np.float32).reshape(-1, 5) for i in ids])
    for tube in linked:
        tube[:, 0] = ids
    # YOWO tube NMS at 0.3, ranked by mean frame confidence.
    ordered = sorted(linked, key=lambda t: -float(t[:, 5].mean()))
    kept = []
    for tube in ordered:
        if all(tube_iou(tube[:, :5], k[:, :5]) <= 0.3 for k in kept):
            kept.append(tube)
    return kept


def video_map(predictions, truths, num_classes, thresholds):
    """Score linked tubes without mutating GT. Tube records: video,label,boxes,score."""
    if num_classes < 1 or not thresholds or any(not 0 < t <= 1 for t in thresholds):
        raise ValueError('invalid class count or IoU thresholds')
    for records in (predictions, truths):
        for r in records:
            if not isinstance(r['label'], (int, np.integer)) or not 0 <= r['label'] < num_classes:
                raise ValueError('tube class outside dataset range')
            tube_array(r['boxes'])
    for r in predictions:
        if not np.isfinite(r['score']) or not 0 <= r['score'] <= 1:
            raise ValueError('tube score must be a finite probability')
    results = {}
    for threshold in thresholds:
        aps, counts = {}, {}
        for label in range(num_classes):
            gt = [r for r in truths if r['label'] == label]
            pred = sorted((r for r in predictions if r['label'] == label), key=lambda r: -r['score'])
            counts[str(label)] = len(gt)
            # Official full splits contain all classes. Synthetic absent classes are excluded.
            if not gt:
                aps[str(label)] = None
                continue
            matched = set()
            tp, fp, pr = 0, 0, [[1., 0.]]
            for r in pred:
                candidates = [(tube_iou(r['boxes'], g['boxes']), i) for i, g in enumerate(gt)
                              if i not in matched and g['video'] == r['video']]
                best = max(candidates, default=(0., -1))
                if best[1] >= 0 and best[0] >= threshold:
                    matched.add(best[1]); tp += 1
                else:
                    fp += 1
                pr.append([tp/(tp+fp), tp/len(gt)])
            aps[str(label)] = float(voc_ap(np.asarray(pr)))
        values = [a for a in aps.values() if a is not None]
        results[str(threshold)] = {'video_map': float(np.mean(values)) if values else None,
                                  'ap_by_class': aps, 'n_gt_by_class': counts,
                                  'evaluated_classes': len(values)}
    return results


def video_name(value):
    p = PurePosixPath(value)
    if p.is_absolute() or len(p.parts) != 2 or '..' in p.parts or '\\' in value:
        raise ValueError(f'invalid class/video name: {value}')
    return p.as_posix()


def official_videos(root, dataset, split):
    root = Path(root)
    if dataset == 'ucf24':
        if split != 1:
            raise ValueError('UCF24 uses official split 1')
        names = [video_name(s.strip()) for s in (root/'splitfiles/testlist01.txt').read_text().splitlines() if s.strip()]
    else:
        if split not in (1, 2, 3):
            raise ValueError('J-HMDB split must be 1, 2 or 3')
        names = []
        for action in CLASSES[dataset]:
            path = root/'splitfiles/splits'/f'{action}_test_split{split}.txt'
            seen = set()
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                name, flag = line.split()
                if flag not in ('1', '2') or name in seen:
                    raise ValueError(f'invalid or duplicate official split row: {line}')
                seen.add(name)
                if flag == '2':
                    names.append(video_name(f'{action}/{Path(name).stem}'))
    if not names or len(names) != len(set(names)):
        raise ValueError('official test list is empty or contains duplicates')
    return sorted(names)


def frame_manifest(root, videos):
    """Enumerate ALL RGB frames, not only action-positive annotation frames."""
    root = Path(root)
    manifest = {}
    for video in videos:
        paths = sorted((p for p in (root/'rgb-images'/video).iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')), key=lambda p: int(p.stem))
        ids = [int(p.stem) for p in paths]
        if not ids or ids != list(range(1, len(ids)+1)):
            raise ValueError(f'{video}: expected all frames numbered 1..N without gaps')
        manifest[video] = paths
    return manifest


def ground_truth(root, dataset, manifest, annotations=None):
    root = Path(root)
    truths = []
    if dataset == 'ucf24':
        from scipy.io import loadmat
        rows = loadmat(annotations or root/'splitfiles/finalAnnots.mat')['annot'].reshape(-1)
        seen = set()
        for row in rows:
            name = str(row['name'].item())
            if name not in manifest:
                continue
            if name in seen:
                raise ValueError(f'duplicate annotation video: {name}')
            seen.add(name)
            if int(row['num_imgs'].item()) != len(manifest[name]):
                raise ValueError(f'{name}: annotation/RGB frame count differs')
            for t in row['tubes'].reshape(-1):
                start, end = int(t['sf'].item()), int(t['ef'].item())
                label = int(t['class'].item())-1
                # Cast BEFORE adding xy+wh: MAT coordinates can be uint8.
                boxes = np.asarray(t['boxes'], dtype=np.float64).copy()
                if boxes.shape != (end-start+1, 4) or not 1 <= start <= end <= len(manifest[name]):
                    raise ValueError(f'{name}: invalid tube bounds or length')
                boxes[:, 2:] += boxes[:, :2]  # reference finalAnnots xywh -> xyxy
                tube = tube_array(np.column_stack((np.arange(start, end+1), boxes)))
                if not 0 <= label < 24 or CLASSES[dataset][label] != name.split('/')[0]:
                    raise ValueError(f'{name}: annotation class mismatch')
                truths.append({'video': name, 'label': label, 'boxes': tube})
        if seen != set(manifest):
            raise ValueError(f'missing MAT annotations: {sorted(set(manifest)-seen)[:5]}')
    else:
        if annotations:
            raise ValueError('J-HMDB uses per-frame xyxy labels (one annotated actor per video)')
        for name, paths in manifest.items():
            label = CLASSES[dataset].index(name.split('/')[0])
            rows = []
            for image in paths:
                p = root/'labels'/name/(image.stem+'.txt')
                # J-HMDB RGB videos can include frames outside the annotated
                # action interval. Those frames have no label file and are not
                # part of the ground-truth tube; predictions still cover them.
                if not p.is_file():
                    continue
                a = np.loadtxt(p, ndmin=2)
                if a.shape != (1, 5) or not np.isfinite(a).all() or a[0, 0] != label+1:
                    raise ValueError(f'{p}: expected one actor with matching 1-based class')
                rows.append([int(image.stem), *a[0, 1:]])
            if not rows:
                raise ValueError(f'{name}: no J-HMDB action-frame labels found')
            frames = np.asarray(rows)[:, 0].astype(int)
            if len(frames) > 1 and not np.all(np.diff(frames) == 1):
                raise ValueError(f'{name}: J-HMDB labels must form one continuous action interval')
            truths.append({'video': name, 'label': label, 'boxes': tube_array(rows)})
    if not truths:
        raise ValueError('no ground-truth tubes')
    return truths


def predictions_to_tubes(records, manifest, dataset, coordinate_space, input_size=None):
    """Adapt evaluate.py JSON. Require an explicit record for EVERY test frame.

    An empty record means no detections; an omitted record is a coverage error.
    Confidence is used as exported, never re-multiplied by class/actionness.
    """
    from PIL import Image
    if coordinate_space not in ('original', 'input'):
        raise ValueError('coordinate_space must be original or input')
    if coordinate_space == 'input' and (input_size is None or len(input_size) != 2 or min(input_size) <= 0):
        raise ValueError('input coordinates require input_size=(height,width)')
    expected = {f'{v}/{p.name}': (v, int(p.stem), p) for v, paths in manifest.items() for p in paths}
    seen, grouped, sizes = set(), {}, {}
    for r in records:
        name = r['image_id']
        if name not in expected or name in seen:
            raise ValueError(f'extra, wrong-split or duplicate prediction frame: {name}')
        seen.add(name)
        video, frame, path = expected[name]
        boxes = np.asarray(r['boxes'], dtype=np.float64)
        if boxes.shape == (0,):
            boxes = boxes.reshape(0, 4)
        labels, scores = np.asarray(r['labels']), np.asarray(r['scores'], dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.shape != (len(boxes),) or scores.shape != (len(boxes),):
            raise ValueError(f'{name}: inconsistent prediction shapes')
        if not (np.isfinite(boxes).all() and np.isfinite(labels).all() and np.isfinite(scores).all()):
            raise ValueError(f'{name}: nonfinite predictions')
        if (labels != np.floor(labels)).any() or (labels < 0).any() or (labels >= len(CLASSES[dataset])).any():
            raise ValueError(f'{name}: expected zero-based class IDs')
        if (scores < 0).any() or (scores > 1).any() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError(f'{name}: invalid boxes or confidence')
        if video not in sizes:
            with Image.open(path) as image:
                sizes[video] = image.size
        width, height = sizes[video]
        limit_h, limit_w = input_size if coordinate_space == 'input' else (height, width)
        if (boxes < 0).any() or (boxes[:, [0, 2]] > limit_w+1e-4).any() or (boxes[:, [1, 3]] > limit_h+1e-4).any():
            raise ValueError(f'{name}: boxes outside declared coordinate space')
        if coordinate_space == 'input':
            boxes = boxes * np.array([width/limit_w, height/limit_h]*2)
        for label in np.unique(labels).astype(int):
            selected = labels == label
            grouped.setdefault((video, label), {})[frame] = np.column_stack((boxes[selected], scores[selected]))
    if seen != set(expected):
        raise ValueError(f'missing {len(set(expected)-seen)} test frames; include empty prediction records')
    result = []
    for (video, label), detections in sorted(grouped.items()):
        frames = {int(p.stem): detections.get(int(p.stem), np.empty((0, 5))) for p in manifest[video]}
        for tube in link_video(frames):
            result.append({'video': video, 'label': int(label), 'boxes': tube[:, :5], 'score': float(tube[:, 5].mean())})
    return result
