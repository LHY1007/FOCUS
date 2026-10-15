"""Configuration for the Visium HD preprocessing pipeline.

All dataset-level constants live here so that adding a sample means adding one
entry to :data:`SAMPLES` instead of editing every step.

Pipeline (see ``data_preprocessing/README.md`` for the full description)::

    step1_download.py         10x bundle       -> raw_dir
    step2_export_h5ad.py      raw_dir          -> {bin}.h5ad          (optional)
    step3_build_zarr.py       raw_dir          -> zarr_path
    step4_rasterize_bins.py   zarr_path        -> raster_zarr
    step5_extract_he_tiles.py raster_zarr + tissue image -> WSI/
    step6_extract_st_grids.py raster_zarr                 -> HR_ST/, spot_ST/
    step7_stitch_preview.py   WSI/                        -> restored_WSI_256.png
    step8_qc_tile.py          WSI/ + HR_ST/               -> QC figures

Set :data:`ACTIVE_SAMPLE` before running a step.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset root
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/date/hanyu/JYX/data/visiumHD")

# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------
# These constants define the FOCUS patch layout. The dataloader assumes the grid
# sides they imply (256 / 128 / 26); changing one invalidates every tile already
# written.

CROP_MPP = 0.5                      # every slide is rescaled to 0.5 um/px
CROP_WSI_PIXEL = 5120               # tile side, 5120 px = 2560 um of tissue
TILE_STEP = CROP_WSI_PIXEL // 2     # 50 % overlapping grid

SUB_CROP_PIXEL = 320                # sub-patch side for the multi-scale H&E input
RESIZE_TIME = 20                    # 320 / 20 = 16
SMALL_SIDE = SUB_CROP_PIXEL // RESIZE_TIME
PATCH_GRID = SMALL_SIDE * SMALL_SIDE

ST_RELA_WSI_MPP = 20                # 5120 / 20  = 256 bins -> 10 um per bin
SPOT_RELA_WSI_MPP = 200             # 5120 / 200 + 1 = 26 bins -> 100 um per bin

HR_ST_256_SIDE = CROP_WSI_PIXEL // ST_RELA_WSI_MPP
HR_ST_128_SIDE = CROP_WSI_PIXEL // (ST_RELA_WSI_MPP * 2)
SPOT_BINS = CROP_WSI_PIXEL // SPOT_RELA_WSI_MPP + 1

# ---------------------------------------------------------------------------
# H&E sub-patch acceptance
# ---------------------------------------------------------------------------
# A 320 x 320 sub-patch is kept when it has enough texture, is not over-exposed
# and is not a background-only field.

GRAD_THRESH = 20        # Sobel magnitude at or below which a pixel counts as flat
FLAT_RATIO = 0.75       # reject when more than this share of pixels is flat
BRIGHT_THRESH = 220     # reject when mean luminance exceeds this
R_AVG_THRESH = 120      # reject when mean red channel is below this

MIN_SPOTS_PER_TILE = 180    # tile-level gate on the number of in-tissue spots

#: Accumulator dtype for the transcript grids. 10 um bins hold a handful of
#: transcripts; the 20 um HR_ST_128 grid sums four of them. Raise to int32 if you
#: re-bin to a much coarser grid.
COUNT_DTYPE = "int16"

# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
# affine: 2 x 3 registration matrix mapping rasterised 2 um bin coordinates onto
# the full-resolution H&E frame. It is per section and cannot be derived from
# the data; the value below is the matrix used for the manuscript's mouse kidney
# sample.

SAMPLES = {
    "Mouse_Kidney_FFPE": {
        "tissue_image": "Visium_HD_Mouse_Kidney_tissue_image.btf",
        "raw_dir": DATA_ROOT / "Mouse_Kidney",
        "zarr_path": DATA_ROOT / "Mouse_Kidney" / "data.zarr",
        "raster_zarr": DATA_ROOT / "Mouse_Kidney" / "data_with_rasters.zarr",
        "out_root": DATA_ROOT / "Visium_HD_Mouse_Kidney_FFPE",
        "layer": "20240612",
        "affine": [
            [7.30294570e00, -2.03362911e-02, 4.68210838e03],
            [-2.04357117e-02, -7.30278299e00, 2.67365002e04],
        ],
    },
    "Human_Breast_Cancer_FFPE": {
        "tissue_image": "Visium_HD_Human_Breast_Cancer_Fresh_Frozen_tissue_image.tif",
        "raw_dir": DATA_ROOT / "Human_Breast_Cancer",
        "zarr_path": DATA_ROOT / "Human_Breast_Cancer" / "data.zarr",
        "raster_zarr": DATA_ROOT / "Human_Breast_Cancer" / "data_with_rasters.zarr",
        "out_root": DATA_ROOT / "Visium_HD_Human_Breast_Cancer",
        "layer": "20250725",
        # Placeholder: estimate from your own registration before use.
        "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    },
}

ACTIVE_SAMPLE = "Mouse_Kidney_FFPE"


def sample(name: str | None = None) -> dict:
    """Return the configuration for *name* (default: :data:`ACTIVE_SAMPLE`).

    Paths are expanded to :class:`~pathlib.Path`, and the three output roots are
    derived from ``out_root`` and ``layer``.
    """
    key = name or ACTIVE_SAMPLE
    if key not in SAMPLES:
        raise KeyError(f"unknown sample {key!r}; known: {sorted(SAMPLES)}")

    cfg = dict(SAMPLES[key])
    cfg["name"] = key
    cfg["out_root"] = Path(cfg["out_root"])
    cfg["wsi_root"] = cfg["out_root"] / "WSI" / "extract" / cfg["layer"]
    cfg["hrst_root"] = cfg["out_root"] / "HR_ST" / "extract" / cfg["layer"]
    cfg["spotst_root"] = cfg["out_root"] / "spot_ST" / "extract" / cfg["layer"]
    return cfg
