"""Make a small execution-only demo from existing real Xenium grids and WMB scRNA.

The H&E-derived masks are threshold components, not validated cell segmentation.
scGPT and GigaPath inputs are explicit zero placeholders. Do not use this demo to
assess biological accuracy or reproduce the paper's reported metrics.
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import sparse
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu


def add_patch(src, sample, tile, chosen):
    for modality in ('HR_ST', 'spot_ST', 'WSI'):
        (sample/modality/'extract'/'demo'/tile).mkdir(parents=True, exist_ok=True)
    sparse.save_npz(sample/'HR_ST/extract/demo'/tile/'HR_ST_256.npz',
                    sparse.load_npz(src/'HR_ST'/tile/'HR_ST_256.npz').tocsr()[:, chosen])
    sparse.save_npz(sample/'spot_ST/extract/demo'/tile/'spot_ST.npz',
                    sparse.load_npz(src/'spot_ST'/tile/'spot_ST.npz').tocsr()[:, chosen])
    wsi = sample/'WSI/extract/demo'/tile
    he = np.load(src/'WSI'/tile/'5120_to256.npy')
    subtiles = np.load(src/'WSI'/tile/'320_to16.npy')
    if len(subtiles) < 10:
        raise ValueError(f'Tile {tile} has too few H&E subpatches')
    np.save(wsi/'5120_to256.npy', he)
    np.save(wsi/'320_to16.npy', subtiles)
    hematoxylin = rgb2hed(he)[..., 0]
    labels, _ = ndimage.label(hematoxylin > threshold_otsu(hematoxylin))
    sizes = np.bincount(labels.ravel())
    ids = np.flatnonzero((sizes >= 2) & (sizes <= 256))
    ids = ids[ids != 0]
    ids = ids[np.argsort(-sizes[ids])[:32]]
    if len(ids) < 2:
        raise ValueError(f'Tile {tile} has too few proxy mask components')
    np.save(wsi/'cell_mask.npy', np.stack([labels == i for i in ids], axis=-1).astype(np.uint8))
    np.save(wsi/'pre_he.npy', np.zeros((3, 512), np.float32))
    emb = sample/'spot_ST/extract/demo'/tile/'scgpt_data'
    emb.mkdir(exist_ok=True)
    for g in (1, 3, 5):
        np.save(emb/f'1to{g}.npy', np.zeros((512, 26, 26), np.float32))
    return {'tile': tile, 'he_shape': list(he.shape), 'mask_components': len(ids)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Processed Xenium directory')
    p.add_argument('--reference', type=Path, required=True, help='WMB log2 scRNA h5ad')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tile', default='3_10')
    p.add_argument('--second-tile', default='3_13', help='A second real patch for batch-size-two BatchNorm training')
    a = p.parse_args()
    src, tile = a.source, a.tile
    hr = sparse.load_npz(src/'HR_ST'/tile/'HR_ST_256.npz').tocsr()
    spot = sparse.load_npz(src/'spot_ST'/tile/'spot_ST.npz').tocsr()
    names = pd.read_csv(src/'gene_names.csv').iloc[:, 0].astype(str).to_numpy()
    ref = ad.read_h5ad(a.reference, backed='r')
    symbols = ref.var['gene_symbol'].astype(str).to_numpy()
    common = np.flatnonzero(np.isin(names, symbols))
    totals = np.asarray(hr.sum(axis=0)).ravel()
    chosen = common[np.argsort(-totals[common])[:50]]
    if len(chosen) != 50:
        raise ValueError('Need 50 shared genes for the 5-gene demo memory bank')
    genes = names[chosen]
    lookup = {name: i for i, name in enumerate(symbols)}
    # Read a bounded set of real reference cells and restore the published log2 values.
    with h5py.File(a.reference, 'r') as f:
        matrix = f['X']
        if isinstance(matrix, h5py.Group) and matrix.attrs['encoding-type'] == 'csr_matrix':
            ptr = matrix['indptr'][:1001]
            stop = int(ptr[-1])
            bounded = sparse.csr_matrix((matrix['data'][:stop], matrix['indices'][:stop], ptr),
                                        shape=(len(ptr)-1, ref.n_vars))
            x = bounded[:, [lookup[g] for g in genes]].toarray()
        else:
            raise ValueError('This preparation script expects a CSR WMB log2 reference')
    raw = np.maximum(np.expm1(x.astype(np.float64) * np.log(2)), 0).astype(np.float32)
    raw = raw[raw.sum(axis=1) > 0][:100]
    if len(raw) < 2:
        raise ValueError('Insufficient nonzero reference cells')
    ref.file.close()

    cls = a.output/'Mouse_healthy_brain'
    sample = cls/'Xenium_Mouse_Pup_Preview_379'
    (cls/'sc').mkdir(parents=True, exist_ok=True)
    small_ref = ad.AnnData(sparse.csr_matrix(raw), var=pd.DataFrame(index=genes))
    small_ref.write_h5ad(cls/'sc'/'WMB_MB_100cells.h5ad')
    h = hr[:, chosen]
    sample.mkdir(parents=True, exist_ok=True)
    np.save(sample/'gene_order.npy', np.arange(50))
    np.savetxt(sample/'gene_name.txt', genes, fmt='%s')
    dense = np.log1p(h.toarray())
    coex = np.nan_to_num(np.corrcoef(dense.T), nan=0.0).astype(np.float32)
    np.save(cls/'coexpression_matrix.npy', coex)

    patches = [add_patch(src, sample, t, chosen) for t in (tile, a.second_tile)]

    provenance = {
        'purpose': 'Execution validation only; not a paper benchmark or held-out test',
        'source_directory': str(src), 'tile': tile,
        'source_page': 'https://www.10xgenomics.com/datasets/mouse-pup-preview-data-xenium-mouse-tissue-atlassing-panel-1-standard',
        'reference': str(a.reference), 'reference_transform': '2**log2_expression - 1; then loader normalize_total/log1p',
        'genes': genes.tolist(), 'reference_cells': len(raw),
        'expression_shapes': {'hr': list(h.shape), 'spot': [676, 50]},
        'patches': patches,
        'limitations': ['Whole mouse-pup tile; brain location not independently annotated. Legacy loader class folder is Mouse_healthy_brain.',
                         'Adult WMB midbrain reference is used only to exercise the input interface, not a matched pup reference.',
                         'Hematoxylin threshold components are proxy masks, not validated segmented cells.',
                         'scGPT and GigaPath feature arrays are zero placeholders.',
                         'Coexpression computed from this single tile, not an independent cohort.',
                         'Training and inference use the same two patches; metrics are diagnostic only.']}
    (a.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (a.output/'metadata_prompt.txt').write_text('Xenium spatial transcriptomics of a healthy mouse pup; tissue region unspecified. Execution-only demo with proxy conditioning features.')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
