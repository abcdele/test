import math
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from core import ModelConfig, LossConfig, TrainConfig, MIAActionDetector, DetectionCriterion
from core.alignment import GaussianMixtureMask, EnhancedILA
from core.temporal import ReferenceMamba, BidirectionalSSMBlock, DilatedSpatialTemporalMamba, MITokenFusionGate
from core.utils import aligned_ciou, classwise_nms
from core.head import DecoupledAnchorHead, fuse_actionness_category_logits
from core.assignment import BalancedAssignment
from datasets.list_dataset import ActionClipDataset, collate_clips
from core.optimization import train_one_epoch, WarmupMultiStep
from evaluation_ucf24_jhmdb.eval_results import frame_map

class TinyBackbone(nn.Module):
    out_channels = 16

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=8, padding=1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        y = self.conv(x.reshape(b * t, c, h, w))
        return y.reshape(b, t, *y.shape[1:])

def tiny_config(**kwargs):
    options = dict(num_classes=3, clip_length=4, image_size=32, pretrained=False, predictor_hidden=8, d_state=4, d_conv=3, expand=1.0, mamba_backend='reference', head_dim=16, projection_hidden=24, anchors=(4.0, 8.0, 16.0), head_depth=1)
    options.update(kwargs)
    return ModelConfig(**options)

class CoreTests(unittest.TestCase):

    def test_multistep_warmup_epoch_boundaries_and_resume(self):
        optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=1e-4)
        scheduler = WarmupMultiStep(optimizer, 120, 2, 10)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-5)
        scheduler.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-4)
        # Epoch decay must not depend on skipped optimizer updates.
        for expected in (5e-5, 2.5e-5, 1.25e-5, 6.25e-6, 6.25e-6):
            scheduler.step_epoch()
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected)
        other_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=1e-4)
        restored = WarmupMultiStep(other_optimizer, 120, 2, 10)
        restored.load_state_dict(scheduler.state_dict())
        self.assertEqual(restored.state_dict(), scheduler.state_dict())
        self.assertAlmostEqual(other_optimizer.param_groups[0]['lr'], 6.25e-6)

    def test_interaction_token_gate_has_no_spatial_mask_input(self):
        gate = MITokenFusionGate(16)
        x = torch.randn(2, 3, 4, 16)
        tokens = torch.randn(2, 4, 16, requires_grad=True)
        output = gate(x, tokens)
        self.assertEqual(output.shape, x.shape)
        (output.square().mean()).backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(float(tokens.grad.abs().sum()), 0)

    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)

    def test_gaussian_mask_bounds_and_coordinate_gradient(self):
        points = torch.tensor([[[0.1, 0.2], [-0.4, 0.3]]], requires_grad=True)
        weights = torch.tensor([[0.4, 0.6]], requires_grad=True)
        mask = GaussianMixtureMask(normalization='minmax')(points, weights, 7, 9)
        self.assertAlmostEqual(float(mask.detach().min()), 0.5, places=5)
        self.assertAlmostEqual(float(mask.detach().max()), 1.5, places=5)
        (mask * torch.arange(63).reshape(1, 1, 7, 9)).sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(float(points.grad.abs().sum()), 0)
        self.assertGreater(float(weights.grad.abs().sum()), 0)

    def test_degenerate_mask_identity(self):
        mask = GaussianMixtureMask(normalization='minmax')(torch.zeros(1, 1, 2), torch.ones(1, 1), 1, 1)
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))

    def test_alignment_equations_and_mia_double_gaussian_mask(self):
        layer = EnhancedILA(16, 4, 8).eval()
        x = torch.randn(2, 4, 16, 4, 4)
        y, tokens, aux = layer(x)
        self.assertTrue(torch.equal(y[:, 0], x[:, 0]))
        mask = aux['masks']
        coef = 0.3 * aux['motion_scores'][..., None, None, None]
        expected = (1 - coef) * x[:, 1:] + coef * x[:, 1:] * mask
        self.assertTrue(torch.allclose(y[:, 1:], expected, atol=1e-06))
        expected_tokens = layer.token_projection((x[:, 1:] * mask.square()).mean((-2, -1)))
        self.assertTrue(torch.allclose(tokens, expected_tokens, atol=1e-06))
        self.assertTrue(torch.allclose(aux['point_weights'].sum(-1), torch.ones(2, 3)))

    def test_reference_scan_is_causal(self):
        layer = ReferenceMamba(16, 4, 3, 1.0).eval()
        x = torch.randn(2, 6, 16)
        changed = x.clone()
        changed[:, 4:] += 7
        self.assertTrue(torch.allclose(layer(x)[:, :4], layer(changed)[:, :4], atol=1e-06))

    def test_dilation_preserves_order_and_restores_gradients(self):
        block = BidirectionalSSMBlock(16, tiny_config(bidirectional=False))
        block.norm = nn.Identity()
        block.forward_ssm = nn.Identity()
        for dilation in (1, 2, 3, 9):
            x = torch.randn(2, 7, 16, requires_grad=True)
            y = block(x, dilation)
            self.assertTrue(torch.equal(y, x))
            y.sum().backward()
            self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_bidirectional_projection_and_reverse_alignment(self):
        class Cumulative(nn.Module):
            def forward(self, x):
                return x.cumsum(1)
        block = BidirectionalSSMBlock(16, tiny_config())
        block.norm = nn.Identity()
        block.forward_ssm = Cumulative()
        block.backward_ssm = Cumulative()
        with torch.no_grad():
            block.direction_projection.weight.copy_(torch.cat((torch.eye(16), 2 * torch.eye(16)), 1))
            block.direction_projection.bias.zero_()
        x = torch.randn(2, 5, 16)
        expected = x.cumsum(1) + 2 * x.flip(1).cumsum(1).flip(1)
        self.assertTrue(torch.allclose(block(x), expected))

    def test_interaction_gate_is_applied_at_each_temporal_layer(self):
        class Update(nn.Module):
            def forward(self, x, dilation=1):
                return 2 * x
        class Gate(nn.Module):
            def forward(self, x, tokens):
                return 3 * x
        cfg = tiny_config(num_layers=2, dilations=(1, 2))
        layer = DilatedSpatialTemporalMamba(16, cfg)
        layer.layers = nn.ModuleList([Update(), Update()])
        layer.gates = nn.ModuleList([Gate(), Gate()])
        x = torch.randn(1, 4, 16, 2, 2)
        result = layer(x, torch.randn(1, 3, 16))
        self.assertTrue(torch.allclose(result, 49 * x, atol=1e-5))

    def test_rectangular_anchor_decoding_and_center_clamp(self):
        head = DecoupledAnchorHead(16, 3, ((4., 8.), (8., 12.)), 1, center_clamp=5.)
        anchors = torch.tensor([[10., 20., 4., 8.]])
        self.assertTrue(torch.equal(head.decode(anchors, torch.zeros(1, 4)), torch.tensor([[8., 16., 12., 24.]])))
        decoded = head.decode(anchors, torch.tensor([[2., -2., math.log(2), math.log(.5)]]))
        self.assertTrue(torch.allclose(decoded, torch.tensor([[11., 13., 19., 17.]])))
        self.assertEqual(head.box_pred.kernel_size, (3, 3))
        output = head(torch.randn(1, 16, 2, 3), (32, 48))
        self.assertEqual(output['anchors'].shape, (12, 4))
        self.assertTrue(torch.equal(output['anchors'][0], torch.tensor([8., 8., 4., 8.])))

    def test_matcher_uses_both_sources_and_handles_small_empty_batches(self):
        predicted = torch.tensor([[[18., 18., 22., 22.], [8., 8., 12., 12.]]])
        anchors = torch.tensor([[10., 10., 4., 4.], [20., 20., 4., 4.]])
        target = {'boxes': torch.tensor([[8., 8., 12., 12.]]), 'labels': torch.tensor([0])}
        match = BalancedAssignment(1)(predicted, anchors, [target])[0]
        self.assertEqual(set(match[0].tolist()), {0, 1})
        self.assertEqual(match[1].tolist(), [0, 0])
        empty = {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}
        batch = BalancedAssignment(4)(predicted.expand(2, -1, -1), anchors, [empty, target])
        self.assertEqual(len(batch[0][0]), 0)
        self.assertEqual(len(batch[1][0]), 2)

    def test_normalized_targets_equal_pixel_loss_without_mutation(self):
        output = DecoupledAnchorHead(16, 3, (4., 8., 16.), 1)(torch.randn(1, 16, 4, 4), (32, 32))
        boxes = torch.tensor([[8., 7., 20., 25.]])
        normalized = boxes / 32
        labels = torch.tensor([1])
        pixel = DetectionCriterion(3)(output, [{'boxes': boxes, 'labels': labels}])
        scaled = DetectionCriterion(3, LossConfig(targets_normalized=True))(output, [{'boxes': normalized, 'labels': labels}])
        self.assertTrue(torch.allclose(pixel['total'], scaled['total']))
        self.assertTrue(torch.equal(normalized, boxes / 32))

    def test_token_time_correspondence(self):

        class Identity(nn.Module):

            def forward(self, x, dilation=1):
                return x

        class Capture(nn.Module):

            def forward(self, x, tokens):
                self.tokens = tokens.detach().clone()
                return x
        cfg = tiny_config()
        layer = DilatedSpatialTemporalMamba(16, cfg)
        layer.layers[0] = Identity()
        capture = Capture()
        layer.gates[0] = capture
        x = torch.randn(2, 4, 16, 3, 3)
        tokens = torch.randn(2, 3, 16)
        layer(x, tokens)
        self.assertTrue(torch.equal(capture.tokens, torch.cat((tokens[:, :1], tokens), 1)))

    def test_spatial_positions_are_independent(self):
        layer = DilatedSpatialTemporalMamba(16, tiny_config()).eval()
        x = torch.randn(1, 4, 16, 2, 2)
        changed = x.clone()
        changed[:, :, :, 0, 0] += 3
        tokens = torch.randn(1, 3, 16)
        y, z = (layer(x, tokens), layer(changed, tokens))
        self.assertTrue(torch.allclose(y[:, :, :, 1, 1], z[:, :, :, 1, 1], atol=1e-06))

    def test_chunked_and_dilated_scan(self):
        cfg = tiny_config(num_layers=2, dilations=(1, 3))
        layer = DilatedSpatialTemporalMamba(16, cfg).eval()
        x = torch.randn(2, 4, 16, 2, 3)
        tokens = torch.randn(2, 3, 16)
        y = layer(x, tokens)
        for block in layer.layers:
            block.chunk_size = 3
        z = layer(x, tokens)
        self.assertTrue(torch.allclose(y, z, atol=1e-05))
        z.square().mean().backward()
        self.assertTrue(all((torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)))

    def test_full_model_detection_and_gradients(self):
        model = MIAActionDetector(tiny_config(), TinyBackbone())
        x = torch.randn(2, 4, 3, 32, 32)
        output = model(x, return_aux=True)
        self.assertEqual(output['cls_logits'].shape, (2, 48, 3))
        targets = [{'boxes': torch.tensor([[8.0, 7.0, 20.0, 25.0]]), 'labels': torch.tensor([1])}, {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]
        loss = DetectionCriterion(3)(output, targets)
        self.assertTrue(torch.isfinite(loss['total']))
        loss['total'].backward()
        for module in (model.alignment.point_predictor, model.alignment.motion_estimator, model.alignment.token_projection, model.temporal.layers, model.temporal.gates, model.head.cls_pred, model.head.act_pred, model.head.box_pred):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all((torch.isfinite(g).all() for g in grads)))
            self.assertGreater(sum((float(g.abs().sum()) for g in grads)), 0)
        model.eval()
        predictions = model.predict(x, score_threshold=0.0)
        self.assertEqual(len(predictions), 2)
        self.assertTrue((predictions[0]['boxes'] >= 0).all())
        self.assertTrue((predictions[0]['boxes'] <= 32).all())

    def test_main_model_forward(self):
        model = MIAActionDetector(tiny_config(), TinyBackbone()).eval()
        y = model(torch.randn(2, 4, 3, 32, 32))
        self.assertTrue(torch.isfinite(y['boxes']).all())
        self.assertEqual(y['cls_logits'].shape, (2, 48, 3))

    def test_actionness_probability_product_and_gradients(self):
        c = torch.tensor([[[math.log(0.7 / 0.3), -4.0, 2.0]]], requires_grad=True)
        a = torch.tensor([[[math.log(0.8 / 0.2)]]], requires_grad=True)
        fused = fuse_actionness_category_logits(c, a)
        self.assertTrue(torch.allclose(fused.sigmoid(), c.sigmoid() * a.sigmoid(), atol=1e-07))
        self.assertAlmostEqual(float(fused.sigmoid().detach()[0, 0, 0]), 0.56, places=6)
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        from core.region_loss import sigmoid_focal_loss
        sigmoid_focal_loss(fused, target).sum().backward()
        self.assertTrue(torch.isfinite(c.grad).all() and torch.isfinite(a.grad).all())
        self.assertGreater(float(c.grad.abs().sum()), 0.0)
        self.assertGreater(float(a.grad.abs().sum()), 0.0)

    def test_fusion_extreme_logits_and_half_precision(self):
        c = torch.tensor([-10000.0, -100.0, 0.0, 100.0, 10000.0], requires_grad=True)
        a = c.detach().flip(0).clone().requires_grad_(True)
        fused = fuse_actionness_category_logits(c, a)
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.allclose(fused.sigmoid(), c.sigmoid() * a.sigmoid(), atol=1e-07))
        fused.sum().backward()
        self.assertTrue(torch.isfinite(c.grad).all() and torch.isfinite(a.grad).all())
        z = fuse_actionness_category_logits(c.detach().half(), a.detach().half())
        self.assertEqual(z.dtype, torch.float32)
        self.assertTrue(torch.isfinite(z).all())

    def test_head_fused_scores_and_raw_adapter(self):
        head = DecoupledAnchorHead(16, 3, (4.0, 8.0, 16.0), 1).eval()
        pred = head(torch.randn(2, 16, 4, 4), (32, 32))
        self.assertEqual(pred['actionness_logits'].shape, (2, 48, 1))
        self.assertEqual(pred['category_logits'].shape, (2, 48, 3))
        expected = pred['actionness_logits'].sigmoid() * pred['category_logits'].sigmoid()
        self.assertTrue(torch.allclose(pred['cls_logits'].sigmoid(), expected, atol=1e-07))
        raw = head.as_anchor_tensor(pred).view(2, 3, 8, 4, 4)
        flat = raw.permute(0, 3, 4, 1, 2).reshape(2, 48, 8)
        self.assertTrue(torch.equal(flat[..., 4:5], pred['actionness_logits']))
        self.assertTrue(torch.equal(flat[..., 5:], pred['category_logits']))
        detections = head.postprocess(pred, score_threshold=0.0)
        for batch_id, result in enumerate(detections):
            for label, score in zip(result['labels'], result['scores']):
                self.assertTrue(torch.isclose(expected[batch_id, :, label], score, atol=1e-07).any())

    def test_empty_gt_loss(self):
        model = MIAActionDetector(tiny_config(), TinyBackbone()).eval()
        output = model(torch.randn(1, 4, 3, 32, 32))
        loss = DetectionCriterion(3)(output, [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}])
        self.assertEqual(float(loss['box'].detach()), 0.0)
        self.assertTrue(torch.isfinite(loss['total']))

    def test_dual_candidate_assignment_and_conflict_resolution(self):
        head = DecoupledAnchorHead(16, 3, (4., 8., 16.), 1).eval()
        output = head(torch.randn(1,16,4,4),(32,32))
        boxes = torch.tensor([[9.,9.,12.,12.],[9.,9.,13.,13.],[10.,10.,14.,14.],[10.,10.,15.,15.]])
        targets=[{'boxes':boxes,'labels':torch.tensor([0,1,2,1])}]
        criterion=DetectionCriterion(3)
        result=criterion.matcher(output['boxes'],output['anchors'],targets)[0]
        anchor_ids,target_ids,cost=result
        self.assertTrue(torch.isfinite(cost).all())
        self.assertEqual(set(target_ids.tolist()),{0,1,2,3})
        keys=anchor_ids*len(boxes)+target_ids
        self.assertEqual(len(keys),len(keys.unique()))
        selected,ignored=criterion._resolve(output['boxes'][0],boxes,output['priors'],result)
        self.assertTrue(((selected>=-1)&(selected<len(boxes))).all())
        self.assertFalse(ignored[selected>=0].any())
        first=criterion(output,targets)
        second=criterion(output,targets)
        self.assertTrue(torch.equal(first['total'],second['total']))

    def test_ciou_and_classwise_nms(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        self.assertTrue(torch.allclose(aligned_ciou(boxes, boxes), torch.ones(3), atol=1e-06))
        keep = classwise_nms(boxes, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([0, 1, 0]))
        self.assertEqual(len(keep), 3)
        keep = classwise_nms(boxes, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([0, 0, 0]))
        self.assertEqual(keep.tolist(), [0, 2])

    def test_voc_ap_duplicate_penalty(self):
        gt = {'a': {'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]), 'labels': torch.tensor([0, 0])}}
        pred = {'a': {'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]), 'scores': torch.tensor([0.9, 0.8, 0.7]), 'labels': torch.tensor([0, 0, 0])}}
        result = frame_map(pred, gt, 2)
        self.assertAlmostEqual(result['frame_map'], 5 / 6, places=6)
        self.assertIsNone(result['ap_by_class']['1'])

    def test_dataset_sampling_label_scaling_and_train_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images, labels = (root / 'rgb-images' / 'Act' / 'Video', root / 'labels' / 'Act' / 'Video')
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for frame in range(1, 4):
                Image.new('RGB', (64, 48), (frame * 60, 0, 0)).save(images / f'{frame:05d}.png')
                (labels / f'{frame:05d}.txt').write_text('2 16 12 48 36\n')
            listing = root / 'list.txt'
            listing.write_text('Act/Video/00001.png\nAct/Video/00002.png\nAct/Video/00003.png\n')
            cfg, tc = (tiny_config(), TrainConfig(batch_size=1, workers=0, accumulation_steps=2, amp=False))
            ds = ActionClipDataset(root, listing, cfg, tc, False)
            x, target = ds[0]
            self.assertTrue(torch.equal(x[0], x[-1]))
            self.assertTrue(torch.allclose(target['boxes'], torch.tensor([[8.0, 8.0, 24.0, 24.0]])))
            self.assertEqual(target['labels'].tolist(), [1])
            loader = DataLoader(ds, batch_size=1, collate_fn=collate_clips)
            model = MIAActionDetector(cfg, TinyBackbone())
            criterion = DetectionCriterion(3)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            updates = []

            class Scheduler:

                def step(self):
                    updates.append(1)
            statistics = train_one_epoch(model, criterion, loader, optimizer, torch.device('cpu'), tc, scaler, Scheduler())
            self.assertEqual(len(updates), 2)
            self.assertTrue(math.isfinite(statistics['total']))
            (labels / '00001.txt').unlink()
            with self.assertRaises(FileNotFoundError):
                ds[0]
if __name__ == '__main__':
    unittest.main()
