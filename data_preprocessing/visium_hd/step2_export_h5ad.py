"""Step 2 -- export each binned output into an AnnData ``.h5ad``.

Optional. ``step4_rasterize_bins.py`` works from the Zarr written in step 3, but
the per-bin matrices are easier to inspect, subset and hand to scanpy as
``.h5ad``. This step converts the ``filtered_feature_bc_matrix`` inside every
``square_*um`` folder and attaches the spatial coordinates.

    python step2_export_h5ad.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import io

from config import sample

#: Binned outputs to export, in the order Visium HD writes them.
BIN_SIZES = ["square_002um", "square_008um", "square_016um"]

#: Column names Space Ranger uses for the full-resolution pixel position.
COORD_COLUMNS = ["pxl_col_in_fullres", "pxl_row_in_fullres"]


def load_spatial_coordinates(spatial_dir: Path, barcodes) -> np.ndarray:
    """Return an ``(n_barcodes, 2)`` array of full-resolution pixel positions.

    ``tissue_positions.parquet`` is indexed by barcode; the rows are re-ordered
    to match *barcodes* so the coordinates line up with the expression matrix.
    Older Space Ranger releases write ``tissue_positions.csv`` instead, which is
    handled by the fallback.
    """
    parquet = spatial_dir / "tissue_positions.parquet"
    if parquet.exists():
        table = pd.read_parquet(parquet)
        table.index = table["barcode"]
        missing = [c for c in COORD_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(
                f"{parquet} has no {missing}; expected the Visium HD schema"
            )
        return table.loc[barcodes, COORD_COLUMNS].values

    csv = spatial_dir / "tissue_positions.csv"
    if csv.exists():
        table = pd.read_csv(csv)
        table.index = table["barcode"]
        return table.loc[barcodes, COORD_COLUMNS].values

    print("  no tissue_positions file, using zero coordinates")
    return np.zeros((len(barcodes), 2))


def export_sample(sample_dir: Path, output_path: Path, region: str = "region_0") -> None:
    """Write ``sample_dir`` as an AnnData object at *output_path*."""
    matrix_dir = sample_dir / "filtered_feature_bc_matrix"

    matrix = io.mmread(matrix_dir / "matrix.mtx.gz").tocsr().transpose()
    barcodes = pd.read_csv(matrix_dir / "barcodes.tsv.gz", header=None)[0].values
    features = pd.read_csv(matrix_dir / "features.tsv.gz", sep="\t", header=None)[0].values

    adata = anndata.AnnData(X=matrix)
    adata.obs_names = barcodes
    adata.var_names = features
    adata.obsm["spatial"] = load_spatial_coordinates(sample_dir / "spatial", barcodes)

    slide_id = f"{sample_dir.name}_{region}"
    adata.obs["region"] = pd.Series(region, index=adata.obs_names, dtype="category")
    adata.obs["slide_id"] = pd.Series(slide_id, index=adata.obs_names, dtype="category")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(output_path)
    print(f"  wrote {output_path}  ({adata.n_obs} bins x {adata.n_vars} genes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=None,
                        help="key in config.SAMPLES (default: ACTIVE_SAMPLE)")
    parser.add_argument("--bins", nargs="*", default=BIN_SIZES,
                        help="which binned outputs to export")
    args = parser.parse_args()

    cfg = sample(args.sample)
    raw_dir = Path(cfg["raw_dir"])

    for bin_name in args.bins:
        sample_dir = raw_dir / bin_name
        if not sample_dir.is_dir():
            print(f"  skip {bin_name}: {sample_dir} does not exist")
            continue
        try:
            export_sample(sample_dir, raw_dir / f"{bin_name}.h5ad")
        except Exception as exc:                      # noqa: BLE001 - report and continue
            print(f"  failed {bin_name}: {exc}")


if __name__ == "__main__":
    main()
