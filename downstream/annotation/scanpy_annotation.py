"""Cluster a section and rank markers for annotation.

The annotation procedure used for the PCP (Xenium), HNSCC (Open-ST) and mouse
bone marrow (VisiumHD) sections: highly variable genes, PCA, a kNN graph, UMAP
and Leiden clustering, then Wilcoxon rank-sum marker detection. Labels are then
assigned by hand by matching the ranked markers against canonical lineage and
state markers.

This script produces the clustering and the ranked marker tables. The assignment
itself stays manual, so the output is a table to read, plus the clustered object
with the Leiden labels attached.

    python scanpy_annotation.py --input 16-4-NC_SR.h5ad --name 16-4-NC
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

BASE = "/data_temp/mnt/raid5/qnf/bone_project/project_bone"
OUT_DIR = f"{BASE}/downstream/annotation/output"

#: Clustering parameters. These match the preprocessing used for the trajectory
#: step, so the two agree on the same section.
N_TOP_GENES = 3000
N_COMPS = 50
N_NEIGHBORS = 15
N_PCS = 30
RESOLUTION = 0.8

#: Markers reported per cluster. The manual call needs the top of the list; the
#: tail is kept because a lineage marker sometimes ranks below a state marker.
N_MARKERS = 50


def cluster(adata: ad.AnnData) -> None:
    """Cluster *adata* in place following the annotation recipe."""
    sc.pp.filter_cells(adata, min_genes=5)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=10)

    # Keep the raw counts: normalisation overwrites X, and the marker test should
    # be run on the normalised values but the counts are worth retaining.
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES)
    print(f"  highly variable genes: {int(adata.var['highly_variable'].sum())}")

    sc.tl.pca(adata, n_comps=N_COMPS, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=RESOLUTION)
    print(f"  leiden clusters: {adata.obs['leiden'].nunique()}")


def rank_markers(adata: ad.AnnData, groupby: str = "leiden",
                 n_markers: int = N_MARKERS) -> pd.DataFrame:
    """Wilcoxon rank-sum test per cluster; return the top markers as a table."""
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")
    return sc.get.rank_genes_groups_df(adata, group=None).groupby(
        "group", observed=True).head(n_markers).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True,
                        help="section to annotate, e.g. the FOCUS-enhanced h5ad")
    parser.add_argument("--name", required=True,
                        help="sample name, used for the output file names")
    parser.add_argument("--resolution", type=float, default=RESOLUTION,
                        help="Leiden resolution")
    args = parser.parse_args()

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    print(f"{adata.n_obs} cells x {adata.n_vars} genes")

    cluster(adata)
    markers = rank_markers(adata)

    marker_path = out_dir / f"{args.name}_markers.csv"
    markers.to_csv(marker_path, index=False)
    print(f"wrote {marker_path} ({len(markers)} rows)")

    # The cluster label is what a human annotates; keep it alongside the
    # clustering so the manual call can be merged back in.
    adata.obs[["leiden"]].to_csv(out_dir / f"{args.name}_clusters.csv")
    adata.write_h5ad(out_dir / f"{args.name}_clustered.h5ad")
    print(f"wrote {out_dir / f'{args.name}_clustered.h5ad'}")


if __name__ == "__main__":
    main()
