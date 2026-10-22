"""CPU synthetic regression tests; never load a model or write experiment results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from scipy.io import savemat
from evaluation_ucf24_jhmdb.video_map import (
    CLASSES, THRESHOLDS, tube_iou, link_video, video_map, official_videos,
    frame_manifest, ground_truth, predictions_to_tubes,
)
from evaluate_video import main


def tube(start=1, end=3, box=(0, 0, 10, 10)):
    return np.array([[f, *box] for f in range(start, end+1)], dtype=float)


def record(video='A/v', label=0, score=0.9, boxes=None):
    return {'video': video, 'label': label, 'score': score, 'boxes': tube() if boxes is None else boxes}


class VideoMapTests(unittest.TestCase):
    def test_iou_temporal_inclusive_and_spatial(self):
        self.assertEqual(tube_iou(tube(), tube()), 1)
        self.assertEqual(tube_iou(tube(1, 1), tube(1, 1)), 1)
        self.assertAlmostEqual(tube_iou(tube(1, 3), tube(3, 5)), 1/5)
        self.assertEqual(tube_iou(tube(1, 2), tube(3, 4)), 0)
        self.assertEqual(tube_iou(tube(), tube(box=(30, 30, 40, 40))), 0)
        self.assertAlmostEqual(tube_iou(tube(), tube(box=(5, 0, 15, 10))), 6/16)
        # Mean spatial IoU, not intersection-volume / union-volume.
        a = np.array([[1, 0, 0, 9, 9], [2, 0, 0, 99, 99]])
        b = np.array([[1, 0, 0, 9, 9], [2, 0, 0, 9, 9]])
        self.assertAlmostEqual(tube_iou(a, b), (1+.01)/2)
        for bad in (tube()[::-1], tube()[[0, 2]], [[1.5, 0, 0, 10, 10]]):
            with self.assertRaises(ValueError):
                tube_iou(bad, tube())

    def test_ap_duplicates_misses_and_class_video_separation(self):
        gt = [record(), record(video='A/other')]
        pred = [record(score=.9), record(score=.8), record(video='A/other', score=.7)]
        result = video_map(pred, gt, 2, (.2, .5, .75))
        for r in result.values():
            self.assertAlmostEqual(r['video_map'], 5/6)
            self.assertIsNone(r['ap_by_class']['1'])
        self.assertEqual(video_map([], gt, 2, (.5,))['0.5']['video_map'], 0)
        self.assertEqual(video_map([record(label=1)], gt, 2, (.5,))['0.5']['video_map'], 0)
        self.assertEqual(video_map([record(video='wrong')], gt, 2, (.5,))['0.5']['video_map'], 0)
        self.assertEqual(len(gt), 2)

    def test_threshold_boundary_and_unmatched_instances(self):
        gt = [record(boxes=tube(1, 2))]
        result = video_map([record(boxes=tube(1, 4))], gt, 1, (.1, .2, .5, .75))
        self.assertEqual([r['video_map'] for r in result.values()], [1, 1, 1, 0])
        # Two identical annotated instances can each be matched once.
        self.assertEqual(video_map([record(), record()], [record(), record()], 1, (.5,))['0.5']['video_map'], 1)

    def test_linking_gaps_real_ids_score_order_and_no_mutation(self):
        b = np.array([[0, 0, 10, 10, .9], [30, 30, 40, 40, .8]])
        frames = {7: b.copy(), 8: np.empty((0, 5)), 9: b[::-1].copy()}
        tubes = link_video(frames)
        self.assertEqual(len(tubes), 2)
        np.testing.assert_array_equal(tubes[0][:, 0], [7, 8, 9])
        np.testing.assert_allclose(tubes[0][:, 1:5], [[0, 0, 10, 10]]*3)
        self.assertAlmostEqual(float(tubes[0][:, 5].mean()), .9, places=6)
        np.testing.assert_array_equal(frames[7], b)
        self.assertEqual(frames[8].shape, (0, 5))
        single = link_video({10: b[::-1].copy()})
        self.assertAlmostEqual(float(single[0][0, 5]), .9, places=6)
        self.assertEqual(link_video({1: np.empty((0, 5))}), [])
        self.assertEqual(len(link_video({1: np.vstack([b[:1], b[:1]])})), 1)
        with self.assertRaises(ValueError):
            link_video({1: b, 3: b})

    def make_fixture(self, root, dataset='jhmdb21'):
        action = CLASSES[dataset][0]
        video = f'{action}/v'
        for directory in ('rgb-images', 'labels'):
            (root/directory/video).mkdir(parents=True)
        for f in range(1, 4):
            Image.new('RGB', (320, 240)).save(root/'rgb-images'/video/f'{f:05d}.png')
            (root/'labels'/video/f'{f:05d}.txt').write_text('1 32 24 160 120\n')
        splitdir = root/'splitfiles/splits'
        splitdir.mkdir(parents=True)
        if dataset == 'jhmdb21':
            for split in (1, 2, 3):
                for cls in CLASSES[dataset]:
                    (splitdir/f'{cls}_test_split{split}.txt').write_text('v.avi 2\n' if cls == action else 'train.avi 1\n')
        else:
            (root/'splitfiles/testlist01.txt').write_text(video+'\n')
            tubes = np.empty((1, 1), dtype=[('ef', 'O'), ('sf', 'O'), ('class', 'O'), ('boxes', 'O')])
            tubes[0, 0] = (3, 1, 1, np.array([[32, 24, 128, 96]]*3, dtype=np.uint8))
            annot = np.empty((1, 1), dtype=[('num_imgs', 'O'), ('name', 'O'), ('tubes', 'O')])
            annot[0, 0] = (3, video, tubes)
            savemat(root/'splitfiles/finalAnnots.mat', {'annot': annot})
        records = [{'image_id': f'{video}/{f:05d}.png', 'boxes': [[16, 12, 80, 60]], 'labels': [0], 'scores': [.9]} for f in range(1, 4)]
        return video, records

    def test_adapter_scaling_coverage_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); video, records = self.make_fixture(root)
            manifest = frame_manifest(root, [video])
            tubes = predictions_to_tubes(records[::-1], manifest, 'jhmdb21', 'input', (120, 160))
            np.testing.assert_allclose(tubes[0]['boxes'][:, 1:], [[32, 24, 160, 120]]*3)
            gt = ground_truth(root, 'jhmdb21', manifest)
            self.assertEqual(video_map(tubes, gt, 21, (.5,))['0.5']['video_map'], 1)
            for invalid in (records[:-1], records+[records[0]], [{**records[0], 'image_id': 'other/v/00001.png'}]+records[1:]):
                with self.assertRaises(ValueError):
                    predictions_to_tubes(invalid, manifest, 'jhmdb21', 'input', (120, 160))
            for change in ({'labels': [21]}, {'labels': [.5]}, {'scores': [float('nan')]}, {'scores': [1.1]}, {'boxes': [[0, 0, 161, 30]]}):
                with self.assertRaises(ValueError):
                    predictions_to_tubes([{**records[0], **change}]+records[1:], manifest, 'jhmdb21', 'input', (120, 160))
            empty = [{**r, 'boxes': [], 'labels': [], 'scores': []} for r in records]
            self.assertEqual(predictions_to_tubes(empty, manifest, 'jhmdb21', 'original'), [])
            with self.assertRaises(ValueError):
                predictions_to_tubes(records, manifest, 'jhmdb21', 'input')

    def test_jhmdb_unannotated_video_tail_is_outside_ground_truth_tube(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); video, _ = self.make_fixture(root)
            tail = root/'rgb-images'/video/'00004.png'
            Image.new('RGB', (320, 240)).save(tail)
            manifest = frame_manifest(root, [video])
            gt = ground_truth(root, 'jhmdb21', manifest)
            self.assertEqual(gt[0]['boxes'][[0, -1], 0].tolist(), [1, 3])
            records = [{'image_id': f'{video}/{f:05d}.png', 'boxes': [], 'labels': [], 'scores': []}
                       for f in range(1, 5)]
            self.assertEqual(predictions_to_tubes(records, manifest, 'jhmdb21', 'original'), [])

    def test_official_split_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); video, _ = self.make_fixture(root)
            for split in (1, 2, 3):
                path=root/'splitfiles/splits'/f'brush_hair_test_split{split}.txt'
                path.write_text(f'v{split}.avi 2\ntrain.avi 1\n')
                self.assertEqual(official_videos(root, 'jhmdb21', split), [f'brush_hair/v{split}'])
            with self.assertRaises(ValueError):
                official_videos(root, 'ucf24', 2)

    def test_cli_both_datasets_and_no_overwrite(self):
        for dataset in THRESHOLDS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as d:
                root = Path(d); _, records = self.make_fixture(root, dataset)
                cache=root/'predictions.json'; cache.write_text(json.dumps(records))
                args=['--dataset', dataset, '--split', '1', '--root', str(root), '--predictions', str(cache),
                      '--coordinate-space', 'input', '--input-size', '120', '160', '--output', str(root/'out')]
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args)
                result=json.loads((root/'out/video_metrics.json').read_text())
                self.assertEqual(set(result['metrics']), set(map(str, THRESHOLDS[dataset])))
                self.assertTrue(all(v['video_map'] == 1 for v in result['metrics'].values()))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(args)


if __name__ == '__main__':
    unittest.main()
