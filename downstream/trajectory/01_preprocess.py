"""Preprocess the NC spatial sample and score the bone lineage markers.

Filtering, library-size normalisation, log transform, highly variable gene
selection, PCA, neighbour graph, UMAP and Leiden clustering, followed by marker
scores for LepR-MSC, osteoblast, osteocyte, adipocyte and chondrocyte. The
scores are what the trajectory step uses to choose the root and the lineage
endpoints.
"""
import scanpy as sc
import anndata as ad
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

BASE = '/data_temp/mnt/raid5/qnf/bone_project/project_bone'
OUT = f'{BASE}/downstream/trajectory/output'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)

print("Loading NC spatial data...")
adata = ad.read_h5ad(f'{BASE}/xiao/16-4-NC_SR.h5ad')
print(f"Raw: {adata.shape}")

sc.pp.filter_cells(adata, min_genes=5)
sc.pp.filter_cells(adata, min_counts=1)
print(f"After cell QC: {adata.shape}")

sc.pp.filter_genes(adata, min_cells=10)
print(f"After gene filter: {adata.shape}")

adata.obs['imagecol'] = adata.obs['cx'].values
adata.obs['imagerow'] = adata.obs['cy'].values
adata.layers['counts'] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
print(f"HVGs: {adata.var['highly_variable'].sum()}")

sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8)
print(f"Leiden clusters: {adata.obs['leiden'].nunique()}")

# Cell type scoring (same markers as T2DM)
markers = {
    'LepR_MSC': ['Cxcl12', 'Vcam1', 'Kitl', 'Lepr', 'Adipoq'],
    'Osteoblast': ['Sp7', 'Runx2', 'Alpl', 'Ibsp', 'Col1a1'],
    'Osteocyte': ['Sost', 'Dmp1', 'Mepe'],
    'Adipocyte': ['Adipoq', 'Fabp4', 'Lpl', 'Plin1', 'Pparg'],
    'Chondrocyte': ['Sox9', 'Col2a1', 'Acan'],
}
for ct, genes in markers.items():
    genes_found = [g for g in genes if g in adata.var_names]
    print(f"  {ct}: {len(genes_found)}/{len(genes)} found -> {genes_found}")
    if len(genes_found) >= 2:
        sc.tl.score_genes(adata, gene_list=genes_found, score_name=f'{ct}_score')
    elif len(genes_found) == 1:
        from scipy.sparse import issparse
        gene_idx = adata.var_names.get_loc(genes_found[0])
        if issparse(adata.X):
            adata.obs[f'{ct}_score'] = np.asarray(adata.X[:, gene_idx].todense()).flatten()
        else:
            adata.obs[f'{ct}_score'] = adata.X[:, gene_idx]
    else:
        adata.obs[f'{ct}_score'] = 0.0

print("\nScore stats:")
for ct in markers:
    col = f'{ct}_score'
    print(f"  {ct}: mean={adata.obs[col].mean():.4f}, max={adata.obs[col].max():.4f}, >0={(adata.obs[col]>0).sum()}")

print(f"\nSaving: {OUT}/nc_preprocessed.h5ad")
adata.write_h5ad(f'{OUT}/nc_preprocessed.h5ad')
print(f"Final shape: {adata.shape}")
print("Done!")
