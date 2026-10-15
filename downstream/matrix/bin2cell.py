"""Aggregate Visium HD 2 um bins into cell-level expression matrices.

Each bin is assigned to a segmented nucleus with an STRtree point-in-polygon
query, and the counts of the bins falling inside a polygon are summed into that
cell. Bins that miss every polygon are dropped, as are cells that receive no
bin.

Coordinate convention: cell X is ``pxl_col_in_fullres`` and cell Y is
``pxl_row_in_fullres``, which is the convention the rest of the project uses.

Writes one AnnData object per sample and, optionally, per-marker spatial maps.
"""
import ast
import logging
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import openslide
import pandas as pd
import scipy.io
import scipy.sparse as sp
import shapely
from shapely import points as shp_points
from shapely.geometry.polygon import Polygon
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# paths
BASE = Path("/data_temp/mnt/raid5/qnf/bone_project/project_bone")
SEG_DIR = BASE / "cell_segmentation"
ST_DIR  = BASE / "ST"
HE_DIR  = BASE / "HE_image"
OUT_H5  = BASE / "downstream" / "matrix" / "output" / "h5ad"
OUT_FIG = BASE / "downstream" / "matrix" / "output" / "figures"

SAMPLES = [
    dict(name="16-2-LPS1_SR",  st="AXB-1-2-A1/16-2-LPS1_SR",  he="1-2.svs"),
    dict(name="16-2-T2DM_SR",  st="AXB-1-2-A1/16-2-T2DM_SR",  he="1-2.svs"),
    dict(name="16-4-NC_SR",    st="AXB-1-2-A1/16-4-NC_SR",     he="1-2.svs"),
    dict(name="16-1-LPS1_SR",  st="AXB-2-2-D1/16-1-LPS1_SR",  he="2-2.svs"),
    dict(name="16-3-NC_SR",    st="AXB-2-2-D1/16-3-NC_SR",     he="2-2.svs"),
    dict(name="16-4-T2DM_SR",  st="AXB-2-2-D1/16-4-T2DM_SR",  he="2-2.svs"),
]

MARKERS = ["Ptprc", "Col1a1", "Pecam1", "Cd68", "Runx2", "Spp1"]

# helpers

def load_cell_seg(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = ["cell_id", "cx", "cy", "boundary"]
    df["boundary"] = df["boundary"].apply(ast.literal_eval)
    return df

def build_polygons(seg_df: pd.DataFrame):
    """Return list of shapely Polygons (same order as seg_df rows)."""
    polys = []
    for row in seg_df.itertuples():
        pts = row.boundary
        if len(pts) >= 3:
            polys.append(Polygon(pts))
        else:
            polys.append(None)
    return polys

def load_matrix(matrix_dir: Path):
    """Load 10x filtered_feature_bc_matrix, return (sparse csr, barcodes, genes)."""
    mat = scipy.io.mmread(matrix_dir / "matrix.mtx.gz").T.tocsr()  # cells × genes
    barcodes = pd.read_csv(matrix_dir / "barcodes.tsv.gz", header=None)[0].values
    features = pd.read_csv(matrix_dir / "features.tsv.gz", sep="\t", header=None)
    genes = features[1].values  # gene symbols
    return mat, barcodes, genes

def map_bins_to_cells(tp: pd.DataFrame, polys: list):
    """
    Batch point-in-polygon using shapely 2.x STRtree.
    Returns (cell_assign array of length len(tp), each value = cell row index or -1).
    When a bin falls in multiple cells (overlap), assigns to the first hit.
    """
    valid_idx = np.array([i for i, p in enumerate(polys) if p is not None and p.is_valid])
    valid_polys = [polys[i] for i in valid_idx]

    tree = STRtree(valid_polys)
    bin_pts = shp_points(tp[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values)

    log.info(f"  Querying {len(tp)} bins against {len(valid_polys)} cell polygons …")
    # result: (2, n_matches) — row0=bin_idx, row1=valid_poly_idx
    result = tree.query(bin_pts, predicate="within")

    cell_assign = np.full(len(tp), -1, dtype=np.int32)
    if result.shape[1] > 0:
        bin_idx, poly_idx = result
        # if a bin hits multiple cells, keep first occurrence
        _, first = np.unique(bin_idx, return_index=True)
        cell_assign[bin_idx[first]] = valid_idx[poly_idx[first]]

    return cell_assign

def aggregate_expression(mat, cell_assign, n_cells):
    """
    Aggregate bin expression into cell-level matrix using sparse row summation.
    mat: (n_bins, n_genes) sparse csr
    cell_assign: array of cell indices per bin (-1 = unassigned), length n_bins
    Returns sparse (n_cells, n_genes) csr matrix.
    """
    assigned_mask = cell_assign >= 0
    n_assigned = assigned_mask.sum()
    log.info(f"  Aggregating {n_assigned} assigned bins into {n_cells} cells …")

    # build a (n_cells, n_bins) indicator matrix and multiply
    bin_indices = np.where(assigned_mask)[0]
    cell_indices = cell_assign[bin_indices]

    indicator = sp.coo_matrix(
        (np.ones(len(bin_indices), dtype=np.float32), (cell_indices, bin_indices)),
        shape=(n_cells, mat.shape[0])
    ).tocsr()

    agg = indicator.dot(mat)
    return agg.tocsr()

def load_he_thumbnail(he_path: Path, level: int = 2):
    """Return numpy RGB array from SVS at given pyramid level."""
    slide = openslide.OpenSlide(str(he_path))
    mpp = float(slide.properties.get("openslide.mpp-x", 0.5013))
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    return np.array(img), mpp, slide.level_downsamples[level]

def build_anndata(agg_mat, seg_df, genes, he_path, mpp, sample_name, bin_counts):
    obs = seg_df[["cell_id", "cx", "cy"]].copy()
    obs.index = obs["cell_id"].astype(str)
    obs.index.name = None  # avoid conflict with column name
    obs["n_bins"] = bin_counts
    obs["total_counts"] = np.asarray(agg_mat.sum(axis=1)).ravel()
    obs["n_genes"] = np.asarray((agg_mat > 0).sum(axis=1)).ravel()

    var = pd.DataFrame(index=genes)
    var.index.name = "gene_symbol"

    adata = ad.AnnData(X=agg_mat, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.obsm["spatial"] = seg_df[["cx", "cy"]].values  # X=col, Y=row
    adata.obs["boundary"] = seg_df["boundary"].apply(str).values

    img_arr, _, downsample = load_he_thumbnail(he_path)
    scalef = 1.0 / downsample
    adata.uns["spatial"] = {
        sample_name: {
            "images": {"lowres": img_arr},
            "scalefactors": {"tissue_lowres_scalef": scalef, "spot_diameter_fullres": 4.0},
        }
    }
    adata.uns["mpp"] = mpp
    adata.uns["he_image_path"] = str(he_path)
    adata.uns["sample_name"] = sample_name
    return adata

# visualization

def _prep_seg_df(obs):
    df = obs.copy()
    df["boundary"] = df["boundary"].apply(
        lambda x: x if isinstance(x, list) else ast.literal_eval(x)
    )
    return df

def _build_poly_collection(seg_df, values, cmap, norm):
    """PolyCollection with black edges (0.15 lw) for crisp cell boundaries."""
    from matplotlib.collections import PolyCollection
    import matplotlib.cm as mcm
    polys, vals = [], []
    for i, row in enumerate(seg_df.itertuples()):
        pts = row.boundary if isinstance(row.boundary, list) else ast.literal_eval(row.boundary)
        if len(pts) < 3:
            continue
        polys.append(pts)
        vals.append(values[i])
    fc = mcm.get_cmap(cmap)(norm(np.array(vals, dtype=float)))
    pc = PolyCollection(polys, facecolors=fc, edgecolors="black",
                        linewidths=0.15, alpha=0.9, rasterized=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return pc, sm

def plot_global_umi(adata, out_path):
    from matplotlib.colors import LogNorm
    seg_df = _prep_seg_df(adata.obs)
    vals = adata.obs["total_counts"].values
    pos = vals[vals > 0]
    norm = LogNorm(vmin=1, vmax=np.percentile(pos, 99))

    expressed = vals > 0
    sub = seg_df[expressed].reset_index(drop=True)
    sub_vals = vals[expressed]

    fig, ax = plt.subplots(figsize=(16, 8))
    pc, sm = _build_poly_collection(sub, sub_vals, "viridis", norm)
    ax.add_collection(pc)
    cx = adata.obs["cx"].values; cy = adata.obs["cy"].values
    ax.set_xlim(cx.min() - 50, cx.max() + 50)
    ax.set_ylim(cy.max() + 50, cy.min() - 50)
    ax.set_aspect("equal")
    plt.colorbar(sm, ax=ax, label="Total UMI (log)", shrink=0.6)
    ax.set_title(f"{adata.uns['sample_name']} — Cell UMI Distribution")
    ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_global_ngenes(adata, out_path):
    from matplotlib.colors import LogNorm
    seg_df = _prep_seg_df(adata.obs)
    vals = adata.obs["n_genes"].values
    pos = vals[vals > 0]
    norm = LogNorm(vmin=1, vmax=np.percentile(pos, 99))

    expressed = vals > 0
    sub = seg_df[expressed].reset_index(drop=True)
    sub_vals = vals[expressed]

    fig, ax = plt.subplots(figsize=(16, 8))
    pc, sm = _build_poly_collection(sub, sub_vals, "magma", norm)
    ax.add_collection(pc)
    cx = adata.obs["cx"].values; cy = adata.obs["cy"].values
    ax.set_xlim(cx.min() - 50, cx.max() + 50)
    ax.set_ylim(cy.max() + 50, cy.min() - 50)
    ax.set_aspect("equal")
    plt.colorbar(sm, ax=ax, label="# Genes (log)", shrink=0.6)
    ax.set_title(f"{adata.uns['sample_name']} — Detected Genes per Cell")
    ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_local_zoom(adata, tp_tissue, out_path, roi_um=500):
    """Local zoom: high-res SVS patch + cell polygon outlines + bin dots."""
    from matplotlib.collections import PolyCollection

    cx_all = adata.obs["cx"].values
    cy_all = adata.obs["cy"].values
    cx_mid = int(np.median(cx_all))
    cy_mid = int(np.median(cy_all))

    mpp = adata.uns["mpp"]
    half_px = int(roi_um / mpp / 2)

    roi_x0 = max(0, cx_mid - half_px)
    roi_y0 = max(0, cy_mid - half_px)
    roi_w  = half_px * 2
    roi_h  = half_px * 2

    slide = openslide.OpenSlide(adata.uns["he_image_path"])
    roi_img = slide.read_region((roi_x0, roi_y0), 0, (roi_w, roi_h)).convert("RGB")
    slide.close()
    roi_arr = np.array(roi_img)

    seg_df = _prep_seg_df(adata.obs)
    mask_cell = (
        (cx_all >= roi_x0) & (cx_all <= roi_x0 + roi_w) &
        (cy_all >= roi_y0) & (cy_all <= roi_y0 + roi_h)
    )
    sub_obs = seg_df[mask_cell]

    mask_bin = (
        (tp_tissue["pxl_col_in_fullres"] >= roi_x0) &
        (tp_tissue["pxl_col_in_fullres"] <= roi_x0 + roi_w) &
        (tp_tissue["pxl_row_in_fullres"] >= roi_y0) &
        (tp_tissue["pxl_row_in_fullres"] <= roi_y0 + roi_h)
    )
    sub_bins = tp_tissue[mask_bin]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(roi_arr,
              extent=[roi_x0, roi_x0 + roi_w, roi_y0 + roi_h, roi_y0],
              aspect="equal", origin="upper", zorder=0)

    # filled semi-transparent polygons
    fill_polys = [row.boundary for row in sub_obs.itertuples() if len(row.boundary) >= 3]
    if fill_polys:
        pc_fill = PolyCollection(fill_polys, facecolors="steelblue", edgecolors="none",
                                 alpha=0.25, zorder=1)
        ax.add_collection(pc_fill)
    # crisp outlines
    if fill_polys:
        pc_edge = PolyCollection(fill_polys, facecolors="none", edgecolors="cyan",
                                 linewidths=0.6, alpha=0.9, zorder=2)
        ax.add_collection(pc_edge)

    ax.scatter(sub_bins["pxl_col_in_fullres"], sub_bins["pxl_row_in_fullres"],
               s=1.0, c="tomato", alpha=0.5, linewidths=0, zorder=3)

    ax.set_xlim(roi_x0, roi_x0 + roi_w)
    ax.set_ylim(roi_y0 + roi_h, roi_y0)
    ax.set_aspect("equal")
    ax.set_title(f"{adata.uns['sample_name']} — Local Zoom: Cell Morphology + Bins")
    ax.axis("off")
    legend_elements = [
        mpatches.Patch(facecolor="steelblue", edgecolor="cyan", alpha=0.6, label="Cell polygon"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
                   markersize=6, label="2µm bin"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_marker_genes(adata, out_path):
    """Grid of marker gene spatial expression (polygon colored) with HE background."""
    available = [g for g in MARKERS if g in adata.var_names][:6]
    if not available:
        log.warning("No marker genes found, skipping marker plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = np.array(axes).ravel()

    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as mcm

    seg_df = _prep_seg_df(adata.obs)
    cx = adata.obs["cx"].values; cy = adata.obs["cy"].values
    all_pts = [row.boundary for row in seg_df.itertuples()]

    for idx, gene in enumerate(available):
        ax = axes[idx]
        gene_idx = list(adata.var_names).index(gene)
        expr = np.asarray(adata.X[:, gene_idx].todense()).ravel()
        pos_mask = expr > 0

        # non-expressing cells: light gray
        gray_polys = [all_pts[i] for i in range(len(seg_df)) if not pos_mask[i] and len(all_pts[i]) >= 3]
        if gray_polys:
            pc_gray = PolyCollection(gray_polys, facecolors="lightgray", edgecolors="black",
                                     linewidths=0.1, alpha=0.3, rasterized=True)
            ax.add_collection(pc_gray)

        # expressing cells: colored
        yes_polys = [all_pts[i] for i in range(len(seg_df)) if pos_mask[i] and len(all_pts[i]) >= 3]
        yes_vals  = expr[pos_mask & np.array([len(p) >= 3 for p in all_pts])]
        if yes_polys:
            norm5 = Normalize(vmin=0, vmax=max(1, np.percentile(yes_vals, 95)))
            cmap5 = mcm.get_cmap("Reds")
            fc5 = cmap5(norm5(yes_vals))
            pc_yes = PolyCollection(yes_polys, facecolors=fc5, edgecolors="black",
                                    linewidths=0.15, alpha=0.9, rasterized=True)
            ax.add_collection(pc_yes)
            sm5 = plt.cm.ScalarMappable(cmap="Reds", norm=norm5)
            sm5.set_array([])
            plt.colorbar(sm5, ax=ax, label="expr", shrink=0.5)

        ax.set_xlim(cx.min() - 50, cx.max() + 50)
        ax.set_ylim(cy.max() + 50, cy.min() - 50)
        ax.set_aspect("equal")
        ax.set_title(f"{gene} (n={pos_mask.sum():,})", fontsize=12)
        ax.axis("off")

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.suptitle(f"{adata.uns['sample_name']} — Marker Gene Expression", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# main

def process_sample(s):
    name = s["name"]
    log.info(f"=== Processing {name} ===")

    seg_path = SEG_DIR / f"{name}_cell_data.csv"
    st_path  = ST_DIR / s["st"] / "binned_outputs" / "square_002um"
    he_path  = HE_DIR / s["he"]
    fig_dir  = OUT_FIG / name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. load cell segmentation
    log.info("  Loading cell segmentation …")
    seg_df = load_cell_seg(seg_path)
    n_cells = len(seg_df)
    log.info(f"  {n_cells} cells")

    # 2. build polygons + spatial index
    log.info("  Building polygons …")
    polys = build_polygons(seg_df)

    # 3. load tissue positions (in_tissue only)
    log.info("  Loading tissue positions …")
    tp = pd.read_parquet(st_path / "spatial" / "tissue_positions.parquet")
    tp_tissue = tp[tp["in_tissue"] == 1].reset_index(drop=True)
    log.info(f"  {len(tp_tissue)} in-tissue bins")

    # 4. map bins to cells
    cell_assign = map_bins_to_cells(tp_tissue, polys)
    assigned = (cell_assign >= 0).sum()
    log.info(f"  {assigned} bins assigned ({100*assigned/len(tp_tissue):.1f}%)")

    # 5. load expression matrix
    log.info("  Loading expression matrix …")
    mat_dir = st_path / "filtered_feature_bc_matrix"
    mat, barcodes, genes = load_matrix(mat_dir)

    # align matrix rows to tp_tissue order
    barcode_to_row = {b: i for i, b in enumerate(barcodes)}
    tp_mat_idx = tp_tissue["barcode"].map(barcode_to_row).fillna(-1).astype(int).values
    valid_tp = tp_mat_idx >= 0
    cell_assign_filt = cell_assign.copy()
    cell_assign_filt[~valid_tp] = -1

    # sub-matrix aligned to tp_tissue (only bins in filtered matrix)
    valid_rows = tp_mat_idx[valid_tp].astype(int)
    sub_mat = mat[valid_rows]        # (n_valid_bins, n_genes)
    sub_assign = cell_assign_filt[valid_tp]

    # 6. aggregate
    bin_counts = np.bincount(cell_assign_filt[cell_assign_filt >= 0], minlength=n_cells)
    agg_mat = aggregate_expression(sub_mat, sub_assign, n_cells)

    # 7. build AnnData
    log.info("  Building AnnData …")
    slide = openslide.OpenSlide(str(he_path))
    mpp = float(slide.properties.get("openslide.mpp-x", 0.5013))
    slide.close()
    adata = build_anndata(agg_mat, seg_df, genes, he_path, mpp, name, bin_counts)

    # 8. save h5ad
    h5_path = OUT_H5 / f"{name}.h5ad"
    adata.write_h5ad(h5_path)
    log.info(f"  Saved {h5_path}")

    # 9. visualizations
    log.info("  Generating visualizations …")
    plot_global_umi(adata, fig_dir / "01_global_umi.png")
    plot_global_ngenes(adata, fig_dir / "02_global_ngenes.png")
    plot_local_zoom(adata, tp_tissue, fig_dir / "03_local_zoom.png")
    plot_marker_genes(adata, fig_dir / "04_marker_genes.png")
    log.info(f"  Figures saved to {fig_dir}")

if __name__ == "__main__":
    for s in SAMPLES:
        process_sample(s)
    log.info("All samples done.")
