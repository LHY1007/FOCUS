"""Semantic regression checks. All random arrays below are synthetic test inputs."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from guided_diffusion.manuscript import (reference_projection, pearson_cells,
    cell_consistency, ActiveTaskCoordinator, MorphologyAlignment, ManuscriptConditioning)
from guided_diffusion.morphology import complexity_from_native_labels
from guided_diffusion.splits import read_manifest, assign_patients
from guided_diffusion.finetune import LowRankUpdate, configure_finetuning, merged_state_dict
from guided_diffusion.script_util import sr_create_model_and_diffusion
from guided_diffusion.unet import SparseGeneMemoryAttention


class MathematicalChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_pearson_matches_numpy_and_has_gradient(self):
        x = torch.randn(5, 12, requires_grad=True)
        actual = pearson_cells(x)
        np.testing.assert_allclose(actual.detach(), np.corrcoef(x.detach()), atol=1e-6)
        actual[0, 1].backward()
        self.assertGreater(x.grad.abs().sum(), 0)

    def test_ot_keeps_each_sample_reference(self):
        ref = torch.tensor([[[1., 2.], [1., 2.]], [[3., 5.], [3., 5.]]])
        y = torch.zeros(2, 2, 2, 2, requires_grad=True)
        out = reference_projection(ref, y)
        # Uniform column marginal is 1 / n_spots, as in Eq. 774.
        torch.testing.assert_close(out[0, :, 0, 0], torch.tensor([.25, .5]))
        torch.testing.assert_close(out[1, :, 0, 0], torch.tensor([.75, 1.25]))
        self.assertTrue(torch.isfinite(out).all())

    def test_disabled_experts_do_not_affect_other_samples(self):
        module = ActiveTaskCoordinator(8).eval()
        features = [torch.randn(2, 8, 4, 4) for _ in range(3)]
        active = torch.tensor([[True, False, False], [False, False, False]])
        prompt, scores = torch.zeros(2, 768), torch.ones(2, 3)
        a = module(features, active, prompt, scores)
        features[1] += 1000
        b = module(features, active, prompt, scores)
        torch.testing.assert_close(a, b)
        self.assertEqual(a[1].abs().sum(), 0)

    def test_cellular_curriculum_uses_all_nonempty_tiles_at_inference(self):
        class RecordAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.token_counts = []

            def forward(self, query, key, value, need_weights=False):
                self.token_counts.append(key.shape[1])
                return torch.zeros_like(query), None

        module = ManuscriptConditioning(8, 5)
        recorder = RecordAttention()
        module.cell_fusion = recorder
        tissue = torch.randn(1, 8, 2, 2)
        cellular = torch.randn(1, 4, 8, 1, 1)
        scores = torch.tensor([[[.1, .2], [.3, .4]]])
        module.train()
        module.progress = 0
        module.cellular_fusion(tissue, cellular, scores)
        module.eval()
        module.cellular_fusion(tissue, cellular, scores)
        self.assertEqual(recorder.token_counts, [1, 4])

    def test_cell_consistency_reaches_both_encoders(self):
        he = torch.randn(1, 8, 16, 16, requires_grad=True)
        st = torch.randn(1, 8, 16, 16, requires_grad=True)
        masks = torch.zeros(1, 3, 16, 16)
        masks[0, 0, 1:5, 1:5] = 1
        masks[0, 1, 7:11, 7:11] = 1
        masks[0, 2, 11:15, 1:5] = 1
        score = torch.ones(1, 1, 1)
        zero = cell_consistency(he, he, masks, score)
        self.assertLess(zero.item(), 1e-6)
        loss = cell_consistency(he, st, masks, score)
        loss.sum().backward()
        self.assertGreater(he.grad.abs().sum(), 0)
        self.assertGreater(st.grad.abs().sum(), 0)

    def test_deformation_and_gradient_losses_backpropagate(self):
        module = MorphologyAlignment(8).train()
        a = torch.randn(1, 8, 16, 16, requires_grad=True)
        b = torch.randn(1, 8, 16, 16, requires_grad=True)
        masks = torch.zeros(1, 2, 32, 32)
        masks[:, 0, 3:15, 3:15] = 1
        masks[:, 1, 17:29, 17:29] = 1
        _, loss, _ = module(a, b, masks)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(module.flow[-1].weight.grad.abs().sum(), 0)
        self.assertEqual(set(module.last_components), {'ngf', 'gmc', 'gmsc', 'flow'})

    def test_patient_assignment_and_leak_rejection(self):
        rows = [{'tissue': 'brain', 'sample_id': f's{i}', 'patient_id': f'p{i//2}'} for i in range(20)]
        result = assign_patients(rows, seed=12)
        self.assertEqual(result, assign_patients(rows, seed=12))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'samples.csv'
            def write(rows):
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=('tissue', 'sample_id', 'patient_id', 'split'))
                    writer.writeheader(); writer.writerows(rows)
            write(result)
            self.assertEqual(len(read_manifest(path)), 20)
            result[1]['split'] = 'finetune'
            write(result)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                read_manifest(path)

    def test_lora_starts_as_identity_and_learns(self):
        weight = torch.randn(7, 5)
        module = LowRankUpdate(weight, 2)
        torch.testing.assert_close(module(weight), weight)
        module(weight).square().sum().backward()
        self.assertGreater(module.A.grad.abs().sum(), 0)

    def test_native_cell_labels_supply_shape_size_texture_and_density(self):
        image = np.random.default_rng(7).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[1:4, 1:4] = 1
        labels[4:7, 4:7] = 2
        labels[11:16, 11:15] = 3
        scores = complexity_from_native_labels(image, labels, pixel_size_um=1, grid_size=4)
        self.assertEqual(scores.shape, (4, 4))
        self.assertGreater(scores[0, 0], scores[1, 1])
        self.assertEqual(scores[3, 3], 0)

    def test_gene_memory_topk_uses_fifteen_percent(self):
        memory = torch.stack([torch.arange(20, dtype=torch.float32),
                              torch.arange(20, dtype=torch.float32).square()], dim=1)
        query = torch.tensor([[[1., 0.]]])
        actual = SparseGeneMemoryAttention(sparsity=0.15)(query, memory)[0, 0]
        selected = memory[-3:]
        expected = (torch.softmax(selected[:, 0], dim=0)[:, None] * selected).sum(0)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
