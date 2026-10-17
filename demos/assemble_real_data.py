"""Assemble one real mouse-pup sample for the strict FOCUS data loader.

All inputs are previously observed or exported real artifacts. This script only
copies them and records their provenance; it never creates segmentation or a
co-expression matrix.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

GENES = ('Ttn', 'Car3', 'Des', 'Cav1', 'Myoz2')
CLASS = 'Mouse_pup_muscle'
SAMPLE = 'Preview_379'
TILE = '3_10'
LAYER = 'official'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def assemble(source, features, reference, coexpression, coexpression_genes, output):
    source, features, reference, coexpression = map(Path, (source, features, reference, coexpression))
    coexpression_genes, output = Path(coexpression_genes), Path(output)
    if output.exists():
        raise FileExistsError(output)
    with (source / 'gene_metadata.csv').open(newline='') as stream:
        panel = list(csv.DictReader(stream))
    channels = []
    for gene in GENES:
        found = [int(row['channel']) for row in panel if row['Gene_Name'] == gene and row['feature_type'] == 'gene']
        if len(found) != 1:
            raise ValueError(f'Ambiguous source gene {gene}: {found}')
        channels.append(found[0])
    ref = ad.read_h5ad(reference, backed='r')
    try:
        if any(gene not in ref.var_names for gene in GENES) or ref.n_obs < 256:
            raise ValueError('Reference lacks current genes or 256 independent P0 cells')
    finally:
        ref.file.close()
    matrix = np.load(coexpression, mmap_mode='r')
    names = coexpression_genes.read_text(encoding='utf-8').splitlines()
    if matrix.shape != (len(names), len(names)) or any(g not in names for g in GENES):
        raise ValueError('Existing co-expression prior lacks named rows for current genes')
    scgpt = np.load(features / 'scgpt_audited.npy')
    pre_he = np.load(features / 'pre_he.npy')
    masks = np.load(features / 'cell_mask.npy')
    complexity = np.load(features / 'complexity_grid.npy')
    if (scgpt.shape != (512, 26, 26) or pre_he.shape != (257, 1536) or
            masks.shape[:2] != (256, 256) or complexity.shape != (16, 16) or
            not np.any(scgpt) or not np.any(pre_he)):
        raise ValueError('Authentic feature or label shape failed')
    if (features / 'scgpt_audited.genes.txt').read_text().splitlines() != list(GENES):
        raise ValueError('scGPT source gene order mismatch')
    hr = sparse.load_npz(source / 'HR_ST' / TILE / 'HR_ST_256.npz')
    low = sparse.load_npz(source / 'spot_ST' / TILE / 'spot_ST.npz')
    if not np.array_equal(np.asarray(hr[:, channels].sum(0)), np.asarray(low[:, channels].sum(0))):
        raise ValueError('Registered five-gene count totals differ')
    sample = output / CLASS / SAMPLE
    base = Path('extract') / LAYER / TILE
    copies = {
        source / 'HR_ST' / TILE / 'HR_ST_256.npz': sample / 'HR_ST' / base / 'HR_ST_256.npz',
        source / 'spot_ST' / TILE / 'spot_ST.npz': sample / 'spot_ST' / base / 'spot_ST.npz',
        source / 'WSI' / TILE / '5120_to256.npy': sample / 'WSI' / base / '5120_to256.npy',
        source / 'WSI' / TILE / '320_to16.npy': sample / 'WSI' / base / '320_to16.npy',
        features / 'cell_mask.npy': sample / 'WSI' / base / 'cell_mask.npy',
        features / 'cell_mask.json': sample / 'WSI' / base / 'cell_mask.json',
        features / 'complexity_grid.npy': sample / 'WSI' / base / 'complexity_grid.npy',
        features / 'complexity_grid.json': sample / 'WSI' / base / 'complexity_grid.json',
        features / 'pre_he.npy': sample / 'WSI' / base / 'pre_he.npy',
        features / 'scgpt_audited.npy': sample / 'spot_ST' / base / 'scgpt_data' / '1to5.npy',
        features / 'scgpt_audited.genes.txt': sample / 'spot_ST' / base / 'scgpt_data' / '1to5.genes.txt',
        reference: output / CLASS / 'sc' / 'cao_p0_muscle_256.h5ad',
        coexpression: output / CLASS / 'coexpression_matrix.npy',
        coexpression_genes: output / CLASS / 'coexpression_matrix.genes.txt',
    }
    for origin, dest in copies.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, dest)
    np.save(sample / 'gene_order.npy', np.asarray(channels, dtype=np.int64))
    (sample / 'gene_name.txt').write_text('\n'.join(GENES) + '\n', encoding='utf-8')
    (sample / 'metadata_prompt.txt').write_text(
        'Xenium In Situ gene expression; Mus musculus; one-day-old mouse pup; '
        'registered sagittal H&E section; muscle-enriched field inferred from '
        'histology and muscle-gene expression; normal tissue.\n', encoding='utf-8')
    (output / 'samples.csv').write_text(
        'tissue,sample_id,patient_id,split\n'
        f'{CLASS},{SAMPLE},10x_mouse_pup_preview_379,train\n', encoding='utf-8')
    record = {
        'genes': list(GENES), 'source_channels': channels,
        'sample': 'one 10x donor and one tile, fitting/demo only; no independent test',
        'native_he_sha256': sha256(source / 'WSI' / TILE / '5120.npy'),
        'official_label_source_sha256': json.loads((features / 'cell_mask.json').read_text())['source_label_sha256'],
        'morphology': json.loads((features / 'complexity_grid.json').read_text()),
        'reference': 'Cao et al. independent P0 muscle scRNA, not the same donor',
        'prior': 'existing read-only 50-gene co-expression matrix from mouse-pup tile; not independent cohort',
        'source_sha256': {str(k): sha256(k) for k in copies},
        'assembled_sha256': {str(v.relative_to(output)): sha256(v) for v in copies.values()},
    }
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-processed', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--coexpression', type=Path, required=True)
    parser.add_argument('--coexpression-genes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.source_processed, args.features, args.reference,
                      args.coexpression, args.coexpression_genes, args.output)
    print(json.dumps({'genes': result['genes'], 'source_channels': result['source_channels'],
                      'sample': result['sample']}, indent=2))
