"""Small synthetic fixtures for gene identity, offsets and skipped patch indices."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from anndata import AnnData
from guided_diffusion.load_dataset import Data_all


class DataChecks(unittest.TestCase):
    def test_metadata_uses_each_sample_and_current_gene_group(self):
        import torch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            prompts = {}
            for sample, context in [('s1', 'human kidney CosMx'), ('s2', 'mouse brain Xenium')]:
                folder = Path(temp) / 'tissue' / sample
                folder.mkdir(parents=True)
                (folder / 'metadata_prompt.txt').write_text(context)
            class Tokenizer:
                def __call__(self, prompt, **kwargs):
                    prompts[prompt.split()[-1]] = prompt
                    return {'input_ids': torch.ones((1, 2), dtype=torch.long)}
            class Model:
                def __call__(self, **kwargs):
                    return SimpleNamespace(pooler_output=torch.ones((1, 768)))
            data = Data_all.__new__(Data_all)
            data.data_root = temp
            data.sub_gene_info = {('tissue', 's1'): {'namesN': ['G6', 'G7', 'G8', 'G9', 'G10']},
                                  ('tissue', 's2'): {'namesN': ['Ttn', 'Car3', 'Des', 'Cav1', 'Myoz2']}}
            data.tokenizer, data.model = Tokenizer(), Model()
            data._embed_metadata()
            self.assertEqual(len(data.metadata_features), 2)
            self.assertTrue(prompts['CosMx'].startswith('Target genes: G6, G7, G8, G9, G10.'))
            self.assertIn('mouse brain Xenium', prompts['Xenium'])
            self.assertTrue(prompts['Xenium'].startswith('Target genes: Ttn, Car3, Des, Cav1, Myoz2.'))

    def test_coexpression_rows_follow_selected_gene_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'tissue'
            root.mkdir()
            matrix = np.arange(49, dtype=np.float32).reshape(7, 7)
            np.save(root / 'coexpression_matrix.npy', matrix)
            np.savetxt(root / 'coexpression_matrix.genes.txt',
                       ['g3', 'g1', 'g6', 'g0', 'g4', 'g2', 'g5'], fmt='%s')
            data = Data_all.__new__(Data_all)
            data.selected_classes = ['tissue']
            data._load_coexpression(temp)
            actual = data._coexpression_features('tissue', ['g4', 'g1'])
            import torch
            expected = torch.nn.functional.adaptive_avg_pool1d(
                torch.from_numpy(matrix[[4, 1]])[None], 64).squeeze(0)
            torch.testing.assert_close(actual, expected)
            with self.assertRaisesRegex(ValueError, 'absent'):
                data._coexpression_features('tissue', ['g_missing'])

    def test_second_gene_group_uses_names_instead_of_reference_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / 'tissue' / 'sample'
            base.mkdir(parents=True)
            order = np.arange(9, -1, -1)
            names = np.array([f'g{i}' for i in order])
            np.save(base / 'gene_order.npy', order)
            np.savetxt(base / 'gene_name.txt', names, fmt='%s')
            reference = AnnData(np.arange(30, dtype=np.float32).reshape(3, 10))
            reference.var_names = [f'g{i}' for i in range(10)]
            data = Data_all.__new__(Data_all)
            data.data_root = temp
            data.selected_patches = [('tissue', 'sample', 'layer', 'p')]
            data.gene_start, data.gene_num = 5, 5
            data.sc_reference = {'tissue': reference}
            info = data._load_subdataset_gene_info()[('tissue', 'sample')]
            self.assertEqual(info['namesN'], ['g4', 'g3', 'g2', 'g1', 'g0'])
            np.testing.assert_array_equal(info['idxN'], [4, 3, 2, 1, 0])
            np.testing.assert_array_equal(info['reference'], reference.X[:, [4, 3, 2, 1, 0]])
            reference.var_names = [f'wrong{i}' for i in range(10)]
            with self.assertRaisesRegex(ValueError, 'absent'):
                data._load_subdataset_gene_info()

    def test_missing_patch_cannot_shift_tissue_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / 'valid' / 'sample'
            locations = {key: base / folder / 'extract' / 'layer' / 'p' for key, folder in
                         [('hr', 'HR_ST'), ('spot', 'spot_ST'), ('wsi', 'WSI')]}
            for path in locations.values(): path.mkdir(parents=True)
            sparse.save_npz(locations['hr']/'HR_ST_256.npz', sparse.csr_matrix((65536, 5), dtype=np.float32))
            sparse.save_npz(locations['spot']/'spot_ST.npz', sparse.csr_matrix((676, 5), dtype=np.float32))
            np.save(locations['wsi']/'5120_to256.npy', np.ones((256,256,3),np.uint8))
            np.save(locations['wsi']/'320_to16.npy', np.ones((256,16,16,3),np.uint8))
            np.save(locations['wsi']/'cell_mask.npy', np.ones((256,256,1),np.uint8))
            np.save(locations['wsi']/'pre_he.npy', np.ones((257,1536),np.float32))
            np.save(locations['wsi']/'complexity_grid.npy', np.full((16,16),.25,np.float32))
            features = locations['spot']/'scgpt_data';features.mkdir()
            np.save(features/'1to5.npy', np.ones((512,26,26),np.float32))
            genes = [f'g{i}' for i in range(5)]
            np.savetxt(features/'1to5.genes.txt', genes, fmt='%s')
            (base/'st_resolution.json').write_text(json.dumps({
                'original_st_bin_um': 10.0,
                'source': 'fixture 2560 um field / 256 measured bins'}))
            data = Data_all.__new__(Data_all)
            data.gene_start = 0
            data.gene_num = 5
            data.strict_features = False
            data.selected_patches = [('missing','sample','layer','p'), ('valid','sample','layer','p')]
            data.sub_gene_info = {('valid','sample'):dict(idxN=np.arange(5),namesN=genes,
                                                        reference=np.full((3,5),7,np.float32))}
            data._load_data(temp)
            self.assertEqual(data.selected_patches, [('valid','sample','layer','p')])
            self.assertEqual(len(data.patch_paths), 1)
            self.assertFalse(hasattr(data, 'SR_ST_all'))
            self.assertEqual(data.sample_resolution[('valid','sample')], 10.0)
            self.assertEqual(data.sample_resolution_source[('valid','sample')],
                             'fixture 2560 um field / 256 measured bins')


if __name__ == '__main__':
    unittest.main()
