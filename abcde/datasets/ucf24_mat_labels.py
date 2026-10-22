"""Materialize UCF24 per-frame labels from the corrected finalAnnots.mat."""
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.io import loadmat

from evaluation_ucf24_jhmdb.video_map import CLASSES, video_name
from .frame_lists import official_video_splits


def _canonical_rows(row, video):
    rows = {}
    num_frames = int(row['num_imgs'].item())
    for tube in row['tubes'].reshape(-1):
        start, end = int(tube['sf'].item()), int(tube['ef'].item())
        label = int(tube['class'].item())
        if not 1 <= label <= len(CLASSES['ucf24']) or CLASSES['ucf24'][label - 1] != video.split('/')[0]:
            raise ValueError(f'{video}: MAT class id {label} disagrees with its action directory')
        boxes = np.asarray(tube['boxes'], dtype=np.float64).copy()
        if boxes.shape != (end - start + 1, 4) or not np.isfinite(boxes).all():
            raise ValueError(f'{video}: malformed MAT tube box array')
        if start < 1 or end > num_frames:
            raise ValueError(f'{video}: MAT tube frames outside 1..{num_frames}')
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy, inclusive MAT endpoint.
        boxes -= 1.0  # match the established text-label zero-based xyxy convention.
        if (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError(f'{video}: MAT contains a non-positive box')
        for frame, box in zip(range(start, end + 1), boxes):
            rows.setdefault(frame, []).append(np.r_[label, box])
    if not rows:
        raise ValueError(f'{video}: MAT row contains no action tubes')
    return num_frames, rows


def _read_existing(path):
    raw = path.read_text().strip()
    if not raw:
        return np.empty((0, 5), dtype=np.float64)
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError(f'{path}: expected finite class, x1, y1, x2, y2 rows')
    if len(values) and ((values[:, 0] != np.floor(values[:, 0])).any() or
                        (values[:, 0] < 1).any() or (values[:, 0] > 24).any()):
        raise ValueError(f'{path}: invalid 1-based UCF class id')
    if len(values) and (values[:, 3:] <= values[:, 1:3]).any():
        raise ValueError(f'{path}: non-positive box')
    return values


def _matches_existing(video, canonical, labels_root):
    """Verify all extant label rows occur in MAT; report omissions separately."""
    label_dir = labels_root / video
    files = {int(path.stem): path for path in label_dir.glob('*.txt')} if label_dir.is_dir() else {}
    if len(files) != (len(list(label_dir.glob('*.txt'))) if label_dir.is_dir() else 0):
        raise ValueError(f'{label_dir}: duplicate numeric annotation frame names')
    existing_rows = missing_rows_in_existing = missing_rows_in_missing_files = 0
    incomplete_frames = missing_files = 0
    for frame, expected_rows in canonical.items():
        path = files.get(frame)
        if path is None:
            missing_files += 1
            missing_rows_in_missing_files += len(expected_rows)
            incomplete_frames += 1
            continue
        actual = _read_existing(path)
        existing_rows += len(actual)
        expected = [np.asarray(row, dtype=np.float64) for row in expected_rows]
        for item in actual:
            match = next((index for index, row in enumerate(expected)
                          if np.allclose(item, row, rtol=0, atol=1e-3)), None)
            if match is None:
                raise ValueError(f'{path}: existing annotation row is not present in MAT: {item.tolist()}')
            expected.pop(match)
        omissions = len(expected)
        if omissions:
            incomplete_frames += 1
            missing_rows_in_existing += omissions
    for frame, path in files.items():
        if frame not in canonical and _read_existing(path).size:
            raise ValueError(f'{path}: existing positive label has no corresponding MAT tube')
    return (len(files), existing_rows, incomplete_frames, missing_files,
            missing_rows_in_existing, missing_rows_in_missing_files)


def convert_ucf24_mat(root, output_labels, annotations=None):
    """Write a complete labels tree under a new path, never modifying root/labels."""
    root, output_labels = Path(root), Path(output_labels)
    if output_labels.exists():
        raise FileExistsError(f'refusing to overwrite existing output labels: {output_labels}')
    mat_path = Path(annotations) if annotations else root / 'splitfiles' / 'finalAnnots.mat'
    rows = loadmat(mat_path)['annot'].reshape(-1)
    official_train, official_test = official_video_splits(root, 'ucf24', 1)
    official_set = set(official_train) | set(official_test)
    stage = output_labels.with_name(output_labels.name + '.tmp')
    if stage.exists():
        raise FileExistsError(f'refusing to reuse existing temporary directory: {stage}')
    stage.mkdir(parents=True)
    seen = set()
    stats = {
        'source_mat': str(mat_path), 'videos': 0, 'rgb_frames': 0,
        'positive_frames_from_mat': 0, 'boxes_from_mat': 0,
        'existing_label_files': 0, 'existing_label_rows': 0,
        'positive_frames_with_missing_boxes': 0, 'missing_positive_frame_files': 0,
        'missing_mat_boxes_in_existing_files': 0,
        'missing_mat_boxes_in_missing_files': 0,
        'existing_rows_not_in_mat': 0,
    }
    try:
        for row in rows:
            raw_name = str(row['name'].item())
            video_id = raw_name[:-4] if raw_name.lower().endswith('.avi') else raw_name
            video = video_name(video_id)
            if video in seen:
                raise ValueError(f'duplicate MAT video: {video}')
            seen.add(video)
            num_frames, frame_rows = _canonical_rows(row, video)
            rgb_dir = root / 'rgb-images' / video
            image_paths = {}
            if not rgb_dir.is_dir():
                raise FileNotFoundError(f'missing RGB directory: {rgb_dir}')
            for image in rgb_dir.iterdir():
                if image.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    image_id = int(image.stem)
                    if image_id in image_paths:
                        raise ValueError(f'{rgb_dir}: duplicate frame number {image_id}')
                    image_paths[image_id] = image
            if sorted(image_paths) != list(range(1, num_frames + 1)):
                raise ValueError(f'{video}: MAT num_imgs={num_frames}, RGB frame IDs are not 1..N')

            (label_file_count, existing_rows, incomplete_frames, missing_files,
             missing_in_existing, missing_in_missing) = _matches_existing(
                video, frame_rows, root / 'labels')
            stats['existing_label_files'] += label_file_count
            stats['existing_label_rows'] += existing_rows
            stats['positive_frames_with_missing_boxes'] += incomplete_frames
            stats['missing_positive_frame_files'] += missing_files
            stats['missing_mat_boxes_in_existing_files'] += missing_in_existing
            stats['missing_mat_boxes_in_missing_files'] += missing_in_missing
            stats['videos'] += 1
            stats['rgb_frames'] += num_frames
            stats['positive_frames_from_mat'] += len(frame_rows)
            stats['boxes_from_mat'] += sum(map(len, frame_rows.values()))

            target_dir = stage / video
            target_dir.mkdir(parents=True)
            # Match the repository's existing convention: only action-positive
            # frames have files; unannotated/background frames are omitted.
            for frame, frame_rows_for_id in sorted(frame_rows.items()):
                image = image_paths[frame]
                label_path = target_dir / f'{image.stem}.txt'
                lines = [' '.join(f'{value:g}' for value in record)
                         for record in frame_rows_for_id]
                label_path.write_text('\n'.join(lines) + '\n')
        if not seen:
            raise ValueError(f'{mat_path}: no MAT annotation videos found')
        unexpected = seen - official_set
        missing_test = set(official_test) - seen
        if unexpected or missing_test:
            raise ValueError(
                f'official split/MAT video mismatch: unexpected MAT={sorted(unexpected)[:5]}, '
                f'test missing from MAT={sorted(missing_test)[:5]}')
        existing_label_videos = {
            f'{action.name}/{video.name}'
            for action in (root / 'labels').iterdir() if action.is_dir()
            for video in action.iterdir() if video.is_dir()
        }
        labels_without_mat = existing_label_videos - seen
        if labels_without_mat:
            raise ValueError(f'existing label directories without MAT rows: {sorted(labels_without_mat)[:5]}')
        stats['official_train_videos_without_mat_annotations'] = sorted(set(official_train) - seen)
        os.replace(stage, output_labels)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return stats
