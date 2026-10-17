"""Export one genuine FOCUS condition bundle from the strict formal data path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided_diffusion.load_dataset import load_data

GENES = ('Ttn', 'Car3', 'Des', 'Cav1', 'Myoz2')
FIELDS = ('target_normalized', 'low_res', 'WSI_5120', 'WSI_320',
          'gene_class', 'Gene_index_map', 'metadata_feature', 'scale_gt',
          'co_expression', 'WSI_mask', 'sc', 'scgpt', 'pre_he', 'complexity',
          'original_st_bin_um')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def export(data_root, pre_model_dir, output):
    data_root, pre_model_dir, output = map(Path, (data_root, pre_model_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'{output} must be empty')
    data = load_data(str(data_root), 'Mouse_pup_muscle', 'Train', 10, 5, 5,
                     pre_model_dir=str(pre_model_dir),
                     split_manifest=str(data_root / 'samples.csv'), split='train',
                     reference_cells=256, seed=0, strict_features=True)
    if len(data) != 1 or data.selected_patches[0] != (
            'Mouse_pup_muscle', 'Preview_379', 'official', '3_10'):
        raise ValueError('This reproducible demo needs exactly the registered 3_10 tile')
    sample = data_root / 'Mouse_pup_muscle' / 'Preview_379'
    selected = data.sub_gene_info[('Mouse_pup_muscle', 'Preview_379')]
    if tuple(selected['namesN']) != GENES:
        raise ValueError('Five-gene order mismatch')
    raw_high = sparse.load_npz(sample / 'HR_ST/extract/official/3_10/HR_ST_256.npz')
    raw_low = sparse.load_npz(sample / 'spot_ST/extract/official/3_10/spot_ST.npz')
    columns = selected['idxN']
    observed_hr = raw_high[:, columns].toarray().T.reshape(5, 256, 256).astype(np.int32)
    observed_low = raw_low[:, columns].toarray().T.reshape(5, 26, 26).astype(np.int32)
    if not np.array_equal(observed_hr.sum((1, 2)), observed_low.sum((1, 2))):
        raise ValueError('Observed HR/LR counts disagree')
    values = dict(zip(FIELDS, data[0]))
    values.pop('target_normalized')  # Training target is not a model condition.
    for key, val in values.items():
        if hasattr(val, 'detach'):
            values[key] = val.detach().cpu().numpy()
    he = values.pop('WSI_5120').transpose(1, 2, 0).astype(np.uint8)
    Image.fromarray(he).save(output / '3_10.png')
    np.savez_compressed(output / '3_10.npz', **values,
                        observed_hr_counts=observed_hr, observed_low_counts=observed_low,
                        genes=np.asarray(GENES), source_tile=np.asarray('3_10'))
    reference = data_root / 'Mouse_pup_muscle/sc/cao_p0_muscle_256.h5ad'
    import anndata as ad
    ref = ad.read_h5ad(reference, backed='r')
    try:
        reference_cells = ref.obs_names.tolist()
        donors = ref.obs['donor_id'].value_counts().loc[lambda s: s > 0].to_dict()
    finally:
        ref.file.close()
    record = {
        'kind': 'one real-condition train-tile reconstruction; no independent held-out test',
        'source_page': 'https://www.10xgenomics.com/datasets/mouse-pup-preview-data-xenium-mouse-tissue-atlassing-panel-1-standard',
        'source_sample': 'Xenium_V1_mouse_pup, XOA 1.6.0, one P1 donor, CC BY 4.0',
        'source_tile': {'id': '3_10', 'x_um': 3840, 'y_um': 12800, 'width_um': 2560},
        'formal_assembly_provenance_sha256': sha256(data_root / 'provenance.json'),
        'original_st_resolution': json.loads((sample / 'st_resolution.json').read_text()),
        'genes': list(GENES), 'source_channels': columns.tolist(),
        'observed_counts_per_gene': observed_hr.sum((1, 2)).astype(int).tolist(),
        'reference': {'source': 'Cao et al. P0 muscle scRNA, independent study',
                      'source_collection': '45d5d2c3-bc28-4814-aed6-0bb6f0e11c82',
                      'cells': len(reference_cells), 'donors': donors,
                      'source_cell_ids': reference_cells, 'subset_sha256': sha256(reference)},
        'mask': json.loads((data_root / 'Mouse_pup_muscle/Preview_379/WSI/extract/official/3_10/cell_mask.json').read_text())
                if (data_root / 'Mouse_pup_muscle/Preview_379/WSI/extract/official/3_10/cell_mask.json').exists()
                else 'Official 10x cell-boundary labels; 68 selected cell instances',
        'morphology': json.loads((data_root / 'Mouse_pup_muscle/Preview_379/WSI/extract/official/3_10/complexity_grid.json').read_text()),
        'scgpt': 'official v1, all 153 encoder tensors matched; five mouse genes mapped to human ortholog tokens',
        'gigapath': 'official tile encoder, 257 native-resolution vectors',
        'coexpression': 'read-only existing 50-gene mouse-pup prior from this tile; no independent cohort',
        'bioBERT': 'frozen dmis-lab/biobert-v1.1 using actual sample metadata and current five genes',
        'file_sha256': {name: sha256(output / name) for name in ('3_10.png', '3_10.npz')},
    }
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pre-model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.data_root, args.pre_model_dir, args.output)
    print(json.dumps({'file_sha256': result['file_sha256'],
                      'observed_counts_per_gene': result['observed_counts_per_gene']}, indent=2))
