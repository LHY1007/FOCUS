"""Subsample the scored section and assign the k-means clusters.

Intermediate step between ``01_preprocess.py`` and ``02_psts.py``. The PSTS
trajectory is computed on a 50,000-cell subsample rather than the whole section,
and the k-means labels form the top level of the trajectory hierarchy that
``02_psts.py`` then splits into spatial subclusters.

The random seed is fixed, so the subsample, the clustering and therefore the
trajectory are reproducible.

    python 00_subsample.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

BASE = "/data_temp/mnt/raid5/qnf/bone_project/project_bone"
OUT = f"{BASE}/downstream/trajectory/output"

#: Input written by 01_preprocess.py.
IN_H5AD = f"{OUT}/nc_preprocessed.h5ad"

#: Output consumed by 02_psts.py.
OUT_H5AD = f"{OUT}/nc_psts_trajectory.h5ad"

#: Cells kept from the full section.
N_CELLS = 50_000

#: k-means clusters. These become the coarse nodes the trajectory passes through;
#: too few and the lineage paths collapse, too many and the PAGA graph fragments.
N_CLUSTERS = 12

#: Fixed seed for the subsample and for k-means.
SEED = 0

#: Columns 02_psts.py reads. Anything missing there is a silent NaN downstream,
#: so the check below fails loudly instead.
REQUIRED_OBS = [
    "cx", "cy",
    "LepR_MSC_score", "Osteoblast_score", "Osteocyte_score",
    "Adipocyte_score", "Chondrocyte_score",
]


def check_columns(adata: ad.AnnData, required: list[str]) -> None:
    """Raise with the full list of missing columns rather than one at a time."""
    missing = [name for name in required if name not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"{missing} missing from obs; run 01_preprocess.py first and check "
            f"that it produced the lineage scores"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=N_CELLS,
                        help="number of cells to keep")
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS,
                        help="number of k-means clusters")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="random seed for the subsample and k-means")
    args = parser.parse_args()

    print(f"reading {IN_H5AD}")
    adata = ad.read_h5ad(IN_H5AD)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    check_columns(adata, REQUIRED_OBS)

    # Subsample without replacement, keeping the original order so the file is
    # byte-identical for a given seed.
    rng = np.random.default_rng(args.seed)
    keep = np.sort(rng.choice(adata.n_obs, size=min(args.cells, adata.n_obs),
                              replace=False))
    subset = adata[keep].copy()
    print(f"subsampled {subset.n_obs} cells (seed {args.seed})")

    if "X_pca" not in subset.obsm:
        raise KeyError("obsm['X_pca'] missing; 01_preprocess.py computes the PCA")

    labels = KMeans(n_clusters=args.clusters, random_state=args.seed,
                    n_init=10).fit_predict(subset.obsm["X_pca"])
    subset.obs["kmeans"] = pd.Categorical(labels.astype(str))
    print(f"k-means: {args.clusters} clusters, sizes "
          f"{np.bincount(labels).tolist()}")

    Path(OUT_H5AD).parent.mkdir(parents=True, exist_ok=True)
    subset.write_h5ad(OUT_H5AD)
    print(f"wrote {OUT_H5AD}")


if __name__ == "__main__":
    main()
