"""Synthetic tests for full-RGB coverage and split-preserving preparation."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from scipy.io import savemat

from cfg.defaults import ModelConfig, TrainConfig
from datasets.full_video import FullRGBVideoDataset
from datasets.frame_lists import frame_list_for_videos, generate_frame_lists, official_video_splits
from datasets.list_dataset import ActionClipDataset
from datasets.ucf24_mat_labels import convert_ucf24_mat
from evaluation_ucf24_jhmdb.video_map import CLASSES, frame_manifest
from evaluate_video import main as evaluate_video_main
import predict_video


class DataPreparationTests(unittest.TestCase):
    @staticmethod
    def make_rgb_video(root, video, frame_ids=(1, 2, 3)):
        folder = root / 'rgb-images' / video
        folder.mkdir(parents=True, exist_ok=True)
        for frame in frame_ids:
            Image.new('RGB', (40, 30), (frame * 10, 20, 30)).save(folder / f'{frame:05d}.png')

    def test_full_rgb_dataset_includes_unlabelled_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = 'clap/v'
            self.make_rgb_video(root, video, (1, 2, 3, 4))
            # Deliberately provide no labels directory: inference must be RGB-only.
            manifest = frame_manifest(root, [video])
            dataset = FullRGBVideoDataset(
                root, manifest,
                ModelConfig(num_classes=21, image_size=32, clip_length=2, keyframe=-1),
                TrainConfig(sampling_rate=1),
            )
            self.assertEqual(len(dataset), 4)
            targets = [dataset[i][1] for i in range(len(dataset))]
            self.assertEqual([target['image_id'] for target in targets],
                             [f'{video}/{frame:05d}.png' for frame in range(1, 5)])
            self.assertTrue(all(target['original_hw'] == (30, 40) for target in targets))
            first_clip, _ = dataset[0]
            self.assertEqual(tuple(first_clip.shape), (2, 3, 32, 32))
            torch.testing.assert_close(first_clip[0], first_clip[1])

    def test_ucf_frame_lists_follow_official_video_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'splitfiles').mkdir()
            (root / 'splitfiles' / 'trainlist01.txt').write_text('Basketball/train_v\n')
            (root / 'splitfiles' / 'testlist01.txt').write_text('Basketball/test_v\n')
            for video in ('Basketball/train_v', 'Basketball/test_v'):
                self.make_rgb_video(root, video, (1, 2))
                label_dir = root / 'labels' / video
                label_dir.mkdir(parents=True)
                (label_dir / '00001.txt').write_text('1 0 0 10 10\n')
            train, test = official_video_splits(root, 'ucf24', 1)
            self.assertEqual(train, ['Basketball/train_v'])
            self.assertEqual(test, ['Basketball/test_v'])
            train_list = frame_list_for_videos(root, train)
            test_list = frame_list_for_videos(root, test)
            self.assertEqual(train_list, ['Basketball/train_v/00001.png'])
            self.assertEqual(test_list, ['Basketball/test_v/00001.png'])
            with self.assertRaises(ValueError):
                official_video_splits(root, 'ucf24', 2)

    def test_action_dataset_can_select_alternate_labels_without_changing_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = 'Basketball/v'
            self.make_rgb_video(root, video, (1, 2))
            frame_list = root / 'frame_list.txt'
            frame_list.write_text(f'{video}/00001.png\n')
            model_cfg = ModelConfig(num_classes=24, image_size=32, clip_length=2, keyframe=-1)
            train_cfg = TrainConfig()
            with self.assertRaises(FileNotFoundError):
                ActionClipDataset(root, frame_list, model_cfg, train_cfg)[0]

            alternate = root / 'converted_labels'
            label_dir = alternate / video
            label_dir.mkdir(parents=True)
            (label_dir / '00001.txt').write_text('1 0 0 10 10\n')
            _, target = ActionClipDataset(
                root, frame_list, model_cfg, train_cfg, labels_root=alternate,
            )[0]
            self.assertEqual(target['labels'].tolist(), [0])
            self.assertEqual(tuple(target['boxes'].shape), (1, 4))

    def test_generated_lists_use_all_labelled_official_train_and_official_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / 'splitfiles'
            split_dir.mkdir()
            train_videos = [f'Basketball/train_{i:02d}' for i in range(10)]
            test_video = 'Basketball/official_test'
            (split_dir / 'trainlist01.txt').write_text('\n'.join(train_videos) + '\n')
            (split_dir / 'testlist01.txt').write_text(test_video + '\n')
            for video in [*train_videos, test_video]:
                self.make_rgb_video(root, video, (1,))
                label_dir = root / 'labels' / video
                label_dir.mkdir(parents=True)
                (label_dir / '00001.txt').write_text('1 0 0 10 10\n')

            first = generate_frame_lists(root, 'ucf24', 1)
            self.assertEqual(first['train_videos'], train_videos)
            self.assertNotIn('val', first)
            self.assertEqual(first['test_videos'], [test_video])
            self.assertEqual(len(first['train']), len(train_videos))
            self.assertEqual(first['test'], [f'{test_video}/00001.png'])

    def test_jhmdb_official_splits_are_read_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / 'splitfiles' / 'splits'
            split_dir.mkdir(parents=True)
            for split in (1, 2, 3):
                for index, action in enumerate(CLASSES['jhmdb21']):
                    test_video = f'test_{split}_{index}.avi'
                    train_video = f'train_{split}_{index}.avi'
                    (split_dir / f'{action}_test_split{split}.txt').write_text(
                        f'{train_video} 1\n{test_video} 2\n')
                train, test = official_video_splits(root, 'jhmdb21', split)
                self.assertEqual(len(train), 21)
                self.assertEqual(len(test), 21)
                self.assertIn(f'clap/test_{split}_2', test)
                self.assertNotIn(f'clap/test_{1 if split != 1 else 2}_2', test)

    def test_jhmdb_generated_lists_preserve_each_official_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / 'splitfiles' / 'splits'
            split_dir.mkdir(parents=True)
            for split in (1, 2, 3):
                for index, action in enumerate(CLASSES['jhmdb21']):
                    test_video = f'test_{split}_{index}.avi'
                    train_video = f'train_{split}_{index}.avi'
                    (split_dir / f'{action}_test_split{split}.txt').write_text(
                        f'{train_video} 1\n{test_video} 2\n')
                    for name in (train_video, test_video):
                        video = f'{action}/{Path(name).stem}'
                        self.make_rgb_video(root, video, (1,))
                        label_dir = root / 'labels' / video
                        label_dir.mkdir(parents=True)
                        (label_dir / '00001.txt').write_text('1 0 0 10 10\n')
                lists = generate_frame_lists(root, 'jhmdb21', split)
                self.assertEqual(len(lists['train_videos']), 21)
                self.assertEqual(len(lists['test_videos']), 21)
                self.assertNotIn('val', lists)
                self.assertEqual(len(lists['train']), 21)
                self.assertEqual(len(lists['test']), 21)

    def test_ucf_mat_conversion_is_separate_complete_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = 'Basketball/v'
            self.make_rgb_video(root, video, (1, 2, 3, 4))
            label_dir = root / 'labels' / video
            label_dir.mkdir(parents=True)
            (label_dir / '00001.txt').write_text('1 9 19 39 49\n')

            tube_dtype = [('sf', 'O'), ('ef', 'O'), ('class', 'O'), ('boxes', 'O')]
            tubes = np.empty((1, 1), dtype=tube_dtype)
            tubes[0, 0] = (1, 3, 1, np.array([[10, 20, 30, 30]] * 3, dtype=np.uint8))
            annot_dtype = [('name', 'O'), ('num_imgs', 'O'), ('tubes', 'O')]
            annot = np.empty((1, 1), dtype=annot_dtype)
            # The local corrected MAT stores class/video without an .avi suffix.
            annot[0, 0] = ('Basketball/v', 4, tubes)
            split_dir = root / 'splitfiles'
            split_dir.mkdir()
            mat_path = split_dir / 'finalAnnots.mat'
            savemat(mat_path, {'annot': annot})
            (split_dir / 'trainlist01.txt').write_text('Basketball/train_unannotated\n')
            (split_dir / 'testlist01.txt').write_text('Basketball/v\n')

            output = root / 'prepared_labels'
            stats = convert_ucf24_mat(root, output)
            self.assertEqual(stats['videos'], 1)
            self.assertEqual(stats['rgb_frames'], 4)
            self.assertEqual(stats['positive_frames_from_mat'], 3)
            self.assertEqual(stats['positive_frames_with_missing_boxes'], 2)
            self.assertEqual(stats['official_train_videos_without_mat_annotations'],
                             ['Basketball/train_unannotated'])
            self.assertEqual(stats['missing_positive_frame_files'], 2)
            self.assertEqual(stats['missing_mat_boxes_in_missing_files'], 2)
            self.assertEqual(stats['missing_mat_boxes_in_existing_files'], 0)
            self.assertTrue((output / video / '00002.txt').is_file())
            self.assertEqual((output / video / '00003.txt').read_text(), '1 9 19 39 49\n')
            self.assertFalse((output / video / '00004.txt').exists())
            with self.assertRaises(FileExistsError):
                convert_ucf24_mat(root, output)

    def test_full_prediction_json_connects_to_video_map_without_labels_on_rgb_tail(self):
        class EmptyDetector(torch.nn.Module):
            def predict(self, clips, score_threshold, nms_threshold):
                return [{'boxes': torch.empty((0, 4)), 'labels': torch.empty((0,), dtype=torch.long),
                         'scores': torch.empty((0,))} for _ in range(len(clips))]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = 'brush_hair/v'
            self.make_rgb_video(root, video, (1, 2, 3, 4))
            labels = root / 'labels' / video
            labels.mkdir(parents=True)
            (labels / '00001.txt').write_text('1 2 2 20 20\n')
            (labels / '00002.txt').write_text('1 2 2 20 20\n')
            split_dir = root / 'splitfiles' / 'splits'
            split_dir.mkdir(parents=True)
            for action in CLASSES['jhmdb21']:
                split_rows = 'v.avi 2\ntrain.avi 1\n' if action == 'brush_hair' else 'train.avi 1\n'
                for split in (1, 2, 3):
                    (split_dir / f'{action}_test_split{split}.txt').write_text(split_rows)

            model_cfg = ModelConfig(num_classes=21, image_size=32, clip_length=2, keyframe=-1)
            train_cfg = TrainConfig(batch_size=1, workers=0)
            checkpoint = root / 'checkpoint.pt'
            torch.save({'config': {'model': asdict(model_cfg), 'train': asdict(train_cfg)}, 'model': {}}, checkpoint)
            predictions = root / 'full_predictions.json'
            with patch.object(predict_video, 'build_model', return_value=EmptyDetector()):
                predict_video.main([
                    '--checkpoint', str(checkpoint), '--dataset', 'jhmdb21', '--split', '1',
                    '--root', str(root), '--output', str(predictions), '--device', 'cpu',
                ])
            records = json.loads(predictions.read_text())
            self.assertEqual([row['image_id'] for row in records],
                             [f'{video}/{frame:05d}.png' for frame in range(1, 5)])
            self.assertTrue(all(row['boxes'] == [] and row['labels'] == [] for row in records))
            output = root / 'video_metrics'
            evaluate_video_main([
                '--dataset', 'jhmdb21', '--split', '1', '--root', str(root),
                '--predictions', str(predictions), '--coordinate-space', 'input',
                '--input-size', '32', '32', '--output', str(output),
            ])
            result = json.loads((output / 'video_metrics.json').read_text())
            self.assertEqual(result['num_frames'], 4)
            self.assertEqual(result['num_videos'], 1)
            self.assertEqual(result['num_gt_tubes'], 1)
            self.assertEqual(result['num_predicted_tubes'], 0)


if __name__ == '__main__':
    unittest.main()
