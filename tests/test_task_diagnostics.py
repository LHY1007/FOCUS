"""Controlled input changes for the three task diagnostics and physical scale gate."""
import unittest
import torch
from guided_diffusion.manuscript import ManuscriptConditioning, task_diagnostics, resolution_gate


class TaskDiagnosticChecks(unittest.TestCase):
    def test_physical_resolution_uses_metadata_not_array_size(self):
        size = torch.tensor([8.0, 10.0, 10.001, 100.0])
        self.assertEqual(resolution_gate(size).tolist(), [False, False, True, True])
        for invalid in (torch.tensor([0.]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                resolution_gate(invalid)
        with self.assertRaises(ValueError):
            resolution_gate(None)

    def test_spatial_and_cell_perturbations_change_matching_heads(self):
        torch.set_num_threads(2)
        he = torch.full((1, 3, 256, 256), 0.2)
        he[:, :, :, 128:] = 0.8
        st = torch.full((1, 5, 26, 26), 0.2)
        st[:, :, :, 13:] = 0.8
        mask = torch.zeros((1, 2, 256, 256))
        mask[:, 0, :, :128] = 1
        mask[:, 1, :, 128:] = 1
        reference = torch.ones((1, 3, 5))
        shifted_he = torch.roll(he, shifts=80, dims=-1)
        self.assertTrue(torch.allclose(shifted_he.mean((2, 3)), he.mean((2, 3))))
        base = task_diagnostics(st, he, reference, mask)
        shifted = task_diagnostics(st, shifted_he, reference, mask)
        reduced = task_diagnostics(st * (torch.arange(26)[None, None, :, None] % 3 == 0),
                                   he, reference, mask)
        mixed = task_diagnostics(st.mean((2, 3), keepdim=True).expand_as(st),
                                 he, reference, mask)
        self.assertGreater(shifted[1][0, 0] - base[1][0, 0], 0.5)
        self.assertGreater(reduced[0][0, 0] - base[0][0, 0], 0.5)
        self.assertGreater(mixed[2][0, 0] - base[2][0, 0], 0.5)
        model = ManuscriptConditioning(32, 5)
        self.assertFalse(model.task_scores(st, he, reference, mask)[1].any())
        self.assertTrue(model.task_scores(st, shifted_he, reference, mask)[1][0, 1])
        self.assertTrue(model.task_scores(st * 0, he, reference, mask)[1][0, 0])
        self.assertTrue(model.task_scores(st.mean((2, 3), keepdim=True).expand_as(st),
                                               he, reference, mask)[1][0, 2])


if __name__ == '__main__':
    unittest.main()
