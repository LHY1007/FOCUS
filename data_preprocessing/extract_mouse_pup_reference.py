"""Select a real P0 muscle scRNA reference from Cao et al. CELLxGENE H5AD.

Use the 'Major cell cluster: Muscle cells' dataset from collection
45d5d2c3-bc28-4814-aed6-0bb6f0e11c82. X is normalized in that asset,
so this script explicitly uses its integer raw.X layer.
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

TARGETS = ('Ttn', 'Car3', 'Des', 'Cav1', 'Myoz2')


def extract(source, output, cells=256, seed=0):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    original = ad.read_h5ad(source, backed='r')
    try:
        eligible = np.flatnonzero(
            (original.obs['author_day'].astype(str).to_numpy() == 'P0000') &
            (original.obs['development_stage'].astype(str).to_numpy() == 'Theiler stage 27'))
        if len(eligible) < cells:
            raise ValueError(f'Only {len(eligible)} P0 muscle cells available')
        chosen = np.sort(np.random.default_rng(seed).choice(eligible, cells, replace=False))
        counts = original.layers['raw.X'][chosen]
        if not sparse.issparse(counts):
            counts = sparse.csr_matrix(counts)
        counts = counts.tocsr()
        if counts.shape != (cells, original.n_vars) or not np.allclose(counts.data, np.rint(counts.data)):
            raise ValueError('Expected the full, integer scRNA count matrix')
        var = original.var[['gene_short_name']].copy()
        var['source_ensembl_id'] = original.var_names
        var.index = np.asarray(
            var['gene_short_name'].astype('string').fillna('').str.strip().to_numpy(),
            dtype=object)
        if any(not name for name in var.index):
            raise ValueError('Source has unnamed genes; resolve them before reference export')
        var.index.name = None
        ref = ad.AnnData(X=counts, obs=original.obs.iloc[chosen][
            ['donor_id', 'author_day', 'development_stage', 'cell_type', 'author_cell_type']].copy(),
            var=var)
        ref.var_names_make_unique()
        for gene in TARGETS:
            if gene not in ref.var_names:
                raise ValueError(f'Reference lacks {gene}')
        ref.uns['source'] = 'Cao et al., Nature 2024, doi:10.1038/s41586-024-07069-w'
        ref.uns['cellxgene_collection'] = '45d5d2c3-bc28-4814-aed6-0bb6f0e11c82'
        ref.uns['cellxgene_dataset'] = 'eaa7a0b0-36c9-4ec3-8023-d999eeeed441'
        ref.uns['source_asset'] = 'https://datasets.cellxgene.cziscience.com/d33cc838-a0e3-40f7-ae99-0384b13db313.h5ad'
        ref.uns['selection'] = f'P0000, Theiler stage 27, seed={seed}, cells={cells}, no target-expression filter'
        output.parent.mkdir(parents=True, exist_ok=True)
        ref.write_h5ad(output, compression='gzip')
        return {
            'output': str(output), 'cells': ref.n_obs, 'genes': ref.n_vars,
            'source_eligible_cells': len(eligible),
            'donors': ref.obs['donor_id'].value_counts().loc[lambda s: s > 0].to_dict(),
            'cell_types': ref.obs['cell_type'].value_counts().loc[lambda s: s > 0].to_dict(),
            'target_nonzero_cells': {gene: int((ref[:, gene].X.toarray() > 0).sum()) for gene in TARGETS},
            'selection_seed': seed,
        }
    finally:
        original.file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(extract(args.source, args.output, args.cells, args.seed), indent=2))
