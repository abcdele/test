"""Build frame lists from unchanged official video splits and existing labels."""
from pathlib import Path, PurePosixPath

from evaluation_ucf24_jhmdb.video_map import CLASSES, video_name


def _read_ucf_videos(root, part):
    path = Path(root) / 'splitfiles' / f'{part}list01.txt'
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    result = [video_name(value) for value in values]
    if not result or len(result) != len(set(result)):
        raise ValueError(f'{path}: official {part} video list is empty or duplicated')
    return result


def official_video_splits(root, dataset, split):
    root = Path(root)
    if dataset == 'ucf24':
        if split != 1:
            raise ValueError('UCF24 uses official split 1')
        train = _read_ucf_videos(root, 'train')
        test = _read_ucf_videos(root, 'test')
    elif dataset == 'jhmdb21':
        if split not in (1, 2, 3):
            raise ValueError('J-HMDB split must be 1, 2 or 3')
        train, test = [], []
        for action in CLASSES[dataset]:
            path = root / 'splitfiles' / 'splits' / f'{action}_test_split{split}.txt'
            seen = set()
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                name, flag = line.split()
                if flag not in ('1', '2') or name in seen:
                    raise ValueError(f'{path}: invalid or duplicate row {line!r}')
                seen.add(name)
                video = video_name(f'{action}/{Path(name).stem}')
                (test if flag == '2' else train).append(video)
        if not train or not test or len(train) != len(set(train)) or len(test) != len(set(test)):
            raise ValueError(f'J-HMDB split {split}: empty or duplicate official video list')
    else:
        raise ValueError(f'unknown dataset: {dataset}')
    if set(train) & set(test):
        raise ValueError('official train and test videos overlap')
    return sorted(train), sorted(test)


def frame_list_for_videos(root, videos, labels_root=None, missing_label_videos=None):
    """List only extant annotation files, preserving the historical frame protocol."""
    root = Path(root)
    labels_root = Path(labels_root) if labels_root is not None else root / 'labels'
    result = []
    for video in sorted(videos):
        rgb_dir = root / 'rgb-images' / video
        label_dir = labels_root / video
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f'missing RGB video directory: {rgb_dir}')
        if not label_dir.is_dir():
            if missing_label_videos is None:
                raise FileNotFoundError(f'missing label video directory: {label_dir}')
            missing_label_videos.append(video)
            continue
        images = {}
        for path in rgb_dir.iterdir():
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                frame_id = int(path.stem)
                if frame_id in images:
                    raise ValueError(f'{rgb_dir}: duplicate RGB frame number {frame_id}')
                images[frame_id] = path
        for label in sorted(label_dir.glob('*.txt'), key=lambda path: int(path.stem)):
            frame_id = int(label.stem)
            if frame_id not in images:
                raise FileNotFoundError(f'{label}: no matching RGB frame')
            relative = PurePosixPath(video) / images[frame_id].name
            result.append(relative.as_posix())
    if not result:
        raise ValueError('selected official split has no labeled RGB frames')
    if len(result) != len(set(result)):
        raise ValueError('frame list contains duplicates')
    return result


def generate_frame_lists(root, dataset, split, labels_root=None):
    root = Path(root)
    labels_root = Path(labels_root) if labels_root is not None else root / 'labels'
    official_train_videos, test_videos = official_video_splits(root, dataset, split)
    missing_train_labels = [v for v in official_train_videos if not (labels_root / v).is_dir()]
    missing_test_labels = [v for v in test_videos if not (labels_root / v).is_dir()]
    missing_train_set = set(missing_train_labels)
    usable_train = [v for v in official_train_videos if v not in missing_train_set]
    # Match the original list builder: videos without a labels/<video> tree are
    # reported and omitted rather than assigned fabricated/background labels.
    missing_train_set = set(missing_train_labels)
    train_videos = [v for v in official_train_videos if v not in missing_train_set]
    result = {
        'train': frame_list_for_videos(root, train_videos, labels_root, []),
        'test': frame_list_for_videos(root, test_videos, labels_root, []),
        'train_videos': train_videos,
        'official_train_videos': official_train_videos,
        'test_videos': test_videos,
        'train_videos_without_label_dirs': missing_train_labels,
        'test_videos_without_label_dirs': missing_test_labels,
    }
    return result
