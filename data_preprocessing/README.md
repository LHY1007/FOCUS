# Data Preprocessing

This folder contains the code that converts raw **10x Visium**, **10x Visium HD** and **10x Xenium**
output into the per-patch layout that FOCUS trains and evaluates on. It covers the
**H&E tiling** and the **transcript-to-grid binning** steps; the embedding, segmentation and
co-expression inputs that also feed the model are listed under
[additional model inputs](#7-additional-model-inputs).

```
data_preprocessing/
├── visium/
│   ├── visium.py        # Visium: H&E tiles + spot_ST grid
│   └── utils.py         # h5 matrix reader, gradient magnitude, graph helpers
├── visium_hd/
│   ├── config.py        # sample registry and the shared tile geometry
│   ├── utils.py         # download, tile-grid and block-sum helpers
│   ├── step1_download.py          # 10x bundle -> unpacked raw directory
│   ├── step2_export_h5ad.py       # binned outputs -> per-bin .h5ad
│   ├── step3_build_zarr.py        # raw directory -> SpatialData zarr
│   ├── step4_rasterize_bins.py    # bin tables -> gene x (y, x) rasters
│   ├── step5_extract_he_tiles.py  # H&E tiles + multi-scale sub-patch stack
│   ├── step6_extract_st_grids.py  # transcripts -> HR_ST and spot_ST grids
│   ├── step7_stitch_preview.py    # low-resolution tiles -> slide preview
│   └── step8_qc_tile.py           # per-tile H&E / expression check
├── histology/
│   └── cell_mask.py               # H&E -> nuclei -> non-overlapping cell masks (cell_mask.npy)
├── tests/
│   └── test_preprocessing_fixes.py  # regression checks for the two fixed bugs
└── xenium/
    ├── Xenium_pami.py   # Xenium: aligned H&E tiles + HR_ST and spot_ST grids
    ├── replay_registered_tile.py  # bounded single-tile extraction from raw Xenium files
    ├── utils.py         # OME-TIFF affine warping, gradient magnitude, graph helpers
    └── jsontocsv.py     # Xenium gene_panel.json -> gene_groups.csv
```

### Cell segmentation

[`histology/cell_mask.py`](histology/cell_mask.py) reads a registered native
`5120.npy` H&E tile at 0.5 µm/pixel. It uses the pretrained StarDist2D
`2D_versatile_he` model at its saved default probability and NMS thresholds,
filters small nuclear objects, and expands labels by assigning each reachable
background pixel to its nearest nucleus. The resulting 10 µm occupancy channels
are stored as `cell_mask.npy` with shape `(256,256,M)`; `--draw` writes a preview
at the same grid size. The default minimum nuclear area is 100 native pixels
and expansion distance is 5 µm; both are command-line parameters.

```bash
python -m pip install '.[segmentation]'
python data_preprocessing/histology/cell_mask.py --root /path/to/WSI/extract/layer --all --draw
```

The bundled Xenium example uses registered 10x cell boundaries. The separate
`downstream/matrix/bin2cell_pipeline.py` uses bin2cell segmentation settings.

---

## 1. The output layout

Both pipelines write into one directory per sample, with a `{layer}` folder (the scripts call it a
date, e.g. `20221018` / `20241031`) and one folder per patch, named `{i}_{j}`:

```
{tissue_type}/
├── gene_groups.csv              # the gene panel, shared by the cohort's samples
├── {sample_id}/
│   ├── HR_ST/extract/{layer}/{i}_{j}/
│   │   ├── HR_ST_256.npz        # (65536, G) sparse, 256x256 grid at 10 um/bin
│   │   └── HR_ST_128.npz        # (16384, G) sparse, 128x128 grid at 20 um/bin   (Xenium, Visium HD)
│   ├── spot_ST/extract/{layer}/{i}_{j}/
│   │   └── spot_ST.npz          # (676, G) sparse, 26x26 grid at 100 um/bin
│   ├── WSI/extract/{layer}/{i}_{j}/
│   │   ├── 5120_to256.npy       # (256, 256, 3) uint8, the tile at 10 um/px
│   │   ├── 320_to16.npy         # (256, 16, 16, 3) uint8, sub-patches at 10 um/px
│   │   ├── 5120.npy             # (5120, 5120, 3) uint8, full-resolution tile (Xenium, Visium HD)
│   │   └── WSI_tile.png         # preview of the tile (Visium HD)
│   ├── gene_order.npy           # produced downstream, see section 7
│   └── gene_name.txt            # produced downstream, see section 7
└── coexpression_matrix.npy      # produced downstream, see section 7
```

The Visium HD branch also writes `WSI/extract/{layer}/gene_names.csv`, the gene-column order its
transcript grids use, and `WSI_tile.png`, a preview of each tile.

`gene_groups.csv` sits at the cohort level because the gene panel is shared by every sample of a
tissue: the Visium script reads `super_root/gene_groups.csv`, the Xenium script the path in its
`gene_groups` variable. The Visium HD branch has no cohort-level panel: it derives its gene order
from the sample's own bin table and records it in `gene_names.csv`.

### Why those numbers

The whole layout follows from the constants all three platforms share:

| Constant | Value | Meaning |
| --- | --- | --- |
| `crop_mpp` | `0.5` | every WSI is rescaled to 0.5 um/pixel before anything else |
| `crop_pixel` | `5120` | one tile is 5120 x 5120 pixels = **2560 x 2560 um** of tissue |
| `crop_pixel / 2` | `2560` | tiles are laid out on a **50 % overlapping** grid, so neighbouring tiles share a border region |
| `sub_crop_pixel` | `320` | sub-patch used for the multi-scale H&E input = 160 x 160 um |
| `resize_time` | `20` | 320 / 20 = **16**, the side of a downsampled sub-patch |
| `st_rela_wsi_mpp` | `20` | `5120 / 20 = 256` bins per tile -> **10 um per bin** |
| `spotst_rela_wsi_mpp` | `200` | `5120 / 200 + 1 = 26` bins per tile -> **100 um per bin** |

So `5120_to256.npy` and `HR_ST_256.npz` describe the *same* 2560 um field at the same 10 um/px
resolution, one as pixels and one as transcript counts; `spot_ST.npz` is the same field aggregated to
a Visium-like 100 um spot pitch, giving those branches a common conditioning grid shape. The formal loader also
requires paired HR ST, H&E, cell masks, morphology and pretrained features, gene
sidecars, physical resolution metadata, an scRNA reference and a co-expression prior;
these platform scripts alone do not complete that input tree.

`HR_ST_128.npz` is the same transcripts re-binned to 128 x 128 (20 um/bin) and is written by the
Xenium and Visium HD branches. The Visium HD branch additionally aligns its tiles to the transcript
raster rather than to the slide, so its tile indices start at the corner of the covered region — see
[section 4.4](#44-how-tiles-are-selected).

### Sparse matrix convention

`HR_ST_*.npz` and `spot_ST.npz` are `scipy.sparse` CSR matrices saved with `scipy.sparse.save_npz`.
The 2-D grid is flattened row-major, so element `(h, w, g)` of the grid is row `h * W + w` and column
`g` of the matrix, where `g` indexes `gene_groups.csv`. Load them with:

```python
import scipy.sparse as sp
m = sp.load_npz("HR_ST_256.npz")            # (65536, G)
grid = m.toarray().reshape(256, 256, -1)    # (h, w, gene)
```

Counts are accumulated in `int16`; bins with more than 32767 transcripts overflow.
Check the count range before re-binning to a coarser grid.

---

## 2. Requirements

```bash
python -m pip install ".[preprocessing]"
```

Run the command from the repository root. Version constraints are listed in `setup.cfg`;
`FOCUS.yaml` records the original Linux environment.

Additional libraries:

- **`duckdb`** — Xenium only. `Xenium_pami.py` pushes the per-tile transcript window into DuckDB and
  joins it against the gene panel in SQL, which is what keeps the parquet scan from loading the whole
  `transcripts.parquet` into memory.
- **`fsspec`, `pyarrow`, `tifffile`** — the single-tile Xenium command reads coordinate-filtered
  Parquet row groups and compressed H&E TIFF tiles from local files or HTTP range sources.
- **`pyometiff`** — Xenium only, for reading the `.ome.tif` morphology image.
- **`torch` / `anndata` / `scikit-learn` / `matplotlib`** are imported by both `utils.py` files. The
  functions that need them (graph construction, LSI, CLR normalisation) are not used by the
  preprocessing entry points, but the imports are unconditional — install them or trim the imports.
- **`h5py`** is used by `get_matrix_from_h5` in both `utils.py` files; the original `pytables`
  implementation is kept commented out just above it.
- **`spatialdata` / `spatialdata-io`** — Visium HD only, for the Zarr-backed `SpatialData` object and
  the `visium_hd` reader. Pulls in `zarr`, `xarray` and `dask`.
- **`pyvips`** — Visium HD only. The tissue image is a multi-gigabyte pyramidal `.btf`; `pyvips`
  reads single tiles from it on demand instead of materialising the whole slide. The preprocessing
  extra includes `pyvips[binary]`, which supplies libvips. A system libvips installation is an alternative.
- `PIL.Image.MAX_IMAGE_PIXELS` and `OPENCV_IO_MAX_IMAGE_PIXELS` are raised in `utils.py`; whole-slide
  images are far larger than Pillow's default decompression-bomb limit.

The Visium and Xenium scripts are plain entry points driven by constants at the top of the file
rather than by CLI arguments: edit the paths, then run the file. The Visium HD steps read their paths
from `visium_hd/config.py` and accept an optional `--sample` argument, so switching sample is a
one-line edit there.

---

## 3. Visium (`visium/visium.py`)

### 3.1 What it does

```
filtered_feature_bc_matrix.h5 ─┐
tissue_positions.csv ──────────┼─► WSI tile (2560 um, 50% overlap) ─► 5120_to256.npy
image.tif ─────────────────────┘                                  └─► 320_to16.npy
                                 spot barcodes + counts ───────────► spot_ST.npz
```

### 3.2 Inputs

Everything is read from `root`, the Space Ranger `outs/`-style sample directory:

| File | Used for |
| --- | --- |
| `Visium_Fresh_Frozen_Adult_Mouse_Brain_image.tif` (`ori_HE_name`) | the H&E image |
| `spatial/scalefactors_json.json` | `spot_diameter_fullres`, to derive the slide's um/pixel |
| `spatial/tissue_positions.csv` (or `tissue_positions_list.csv`) | barcode -> `in_tissue`, full-res pixel row/col |
| `Visium_Fresh_Frozen_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5` (`feature_matrix`) | the spot x gene count matrix |

The physical pixel size is not read from metadata; it is derived from the spot diameter, using the
Visium convention that a spot is 65 um across:

```python
MPP = 65.0 / json_data['spot_diameter_fullres']
```

The WSI is then rescaled from `MPP` to `crop_mpp = 0.5` um/px, and is kept **transposed** as
`[width, height, 3]` for the rest of the script.

### 3.3 Configuration

```python
super_root     = r'/date/qnf/VisiumHD/visium/Visium_polya_mousebrain/'   # outputs + gene_groups.csv
root           = r'/date/qnf/VisiumHD/visium/Visium_polya_mousebrain'    # raw Space Ranger inputs
ori_HE_name    = 'Visium_Fresh_Frozen_Adult_Mouse_Brain_image.tif'
out_fileName   = '20221018'          # the {layer} folder name; see the header note below
Gene_num       = 32285               # genes to keep, taken from the top of the feature list
feature_matrix = 'Visium_Fresh_Frozen_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5'
```

`out_fileName` is overloaded: it is both the output `{layer}` folder **and** the Space Ranger version
switch for which position file to read. `out_fileName == '20221018'` selects the newer
`tissue_positions.csv`, anything else selects the older `tissue_positions_list.csv`. The accompanying
`header` flag is effectively always `0` — the comparison against `'202210181'` in the source is a typo
for `'20221018'`, so the header-less branch is never taken. If your Space Ranger output is the older
header-less `tissue_positions_list.csv`, fix that line to `header = None` as well.

The gene panel is read from `super_root/gene_groups.csv`, and **written from the h5 feature list if it
does not exist yet**. That auto-generated file is the full `Gene_num`-gene panel; if you want a
specific panel, create `gene_groups.csv` yourself before the first run.

### 3.4 How tiles are selected

The two `Pool` workers walk the same overlapping grid but apply different acceptance rules:

**WSI tiles** (`process_wsi_patch`) are kept when the tile contains **more than 180 in-tissue spots**.
Then, inside the tile, each 320 x 320 sub-patch must pass a tissue filter:

- fewer than 75 % of its pixels may have a Sobel gradient magnitude <= 20 (rejects flat background), **and**
- mean brightness < 220 **and** mean red channel > 120 (rejects over-exposed and non-H&E regions).

Accepted sub-patches are resized to 16 x 16, repeated to fill a 256-entry grid, and sorted by ascending
Shannon entropy before being saved as `320_to16.npy` with shape `(256, 16, 16, 3)`.

**Spot tiles** (`process_spot_patch`) use the same > 180 spot rule, then bin every in-tissue barcode
into a 26 x 26 grid: `x = int((w_spot - i*2560) / 200)`, `y = int((h_spot - j*2560) / 200)`, i.e. 200 px
at 0.5 um/px = 100 um per bin. Only `spot_ST.npz` is written — Visium is the *low-resolution* branch,
so there is no `HR_ST_256.npz` here.

### 3.5 Run

```bash
cd data_preprocessing/visium
python visium.py
```

Two progress bars appear: `Processing WSI` then `Processing Spot-ST`. Both loops are
`imap_unordered` over `min(cpu_count(), len(tasks))` workers.

### 3.6 Caveats

- The scale factors are read with the standard library `json` module; an earlier revision used a local
  `json2` helper that had to be present next to the script. Nothing extra needs to be installed now.
- `MPP` is derived from the spot diameter, so it is only correct for a standard 6.5 mm Visium slide.
  For VisiumHD or a custom capture area, set `MPP` directly from the vendor's pixel size instead.
- The WSI loop is parallel and each worker reads from the pre-resized in-memory `im`, which is why the
  whole rescaled slide must fit in RAM. A 0.5 um/px whole slide can be tens of GB; lower `crop_mpp`
  resolution or process the slide in strips if you run out of memory.
- Tiles are laid out on a 50 % overlapping grid, so the same tissue appears in up to four tiles. The
  per-tile spot count (`> 180`) is a *tile-level* gate; individual empty regions inside a kept tile are
  filtered per 320 x 320 sub-patch for the H&E input, but `spot_ST.npz` is written for the whole tile
  regardless.

---

## 4. Visium HD (`visium_hd/`)

Visium HD is handled by its own pipeline rather than by `visium/visium.py`. The reason is the input:
Visium HD is a 2 um barcode grid over a whole section, not a 6.5 mm slide of 65 um spots, so the
transcripts have to be rasterised onto the array grid and registered to the H&E image before any tile
can be cut.

```
10x bundle ─► raw directory ─► data.zarr ─► data_with_rasters.zarr ─┬─► WSI/{i}_{j}/
                step1           step3        step4                 └─► HR_ST/{i}_{j}/
                (step2: per-bin .h5ad)                                    spot_ST/{i}_{j}/
                                                              step5 + step6
```

The eight steps are separate files, each runnable on its own and each skipping work that already
exists on disk:

| Step | Reads | Writes |
| --- | --- | --- |
| `step1_download.py` | 10x CDN | unpacked raw directory |
| `step2_export_h5ad.py` | `square_*um/filtered_feature_bc_matrix` | `square_*um.h5ad` |
| `step3_build_zarr.py` | raw directory | `data.zarr` |
| `step4_rasterize_bins.py` | `data.zarr` | `data_with_rasters.zarr` |
| `step5_extract_he_tiles.py` | raster Zarr + tissue image | `WSI/` |
| `step6_extract_st_grids.py` | raster Zarr | `HR_ST/`, `spot_ST/` |
| `step7_stitch_preview.py` | `WSI/` | `restored_WSI_256.png` |
| `step8_qc_tile.py` | `WSI/` + `HR_ST/` | per-tile QC figure |

### 4.1 What it does

```
tissue_image.btf ──pyvips──► 5120 px H&E tile ─► 5120.npy
                                               ├─► 5120_to256.npy
                                               └─► 320_to16.npy
rasterized_002um ──warpAffine──► per-gene 2 um image ─► block sum ─┬─► HR_ST_256.npz  (10 um)
                                                                   ├─► HR_ST_128.npz  (20 um)
                                                                   └─► spot_ST.npz    (100 um)
```

### 4.2 Inputs

| File | Used for |
| --- | --- |
| `Visium_HD_<sample>_tissue_image.btf` | the pyramidal H&E image, read with `pyvips` |
| `scalefactors_json.json` | `microns_per_pixel`, to derive the rescale factor to 0.5 um/px |
| `square_*um/` | the binned count matrices, one per bin size |
| `data_with_rasters.zarr` | the gene rasters, from step 4 |
| the registration matrix | see 4.3 |

`microns_per_pixel` is read from `scalefactors_json.json`. Note that Visium HD publishes it as an
absolute µm/px value, unlike Visium where the pixel size has to be derived from the 65 um spot
diameter.

### 4.3 Configuration

All dataset-level constants live in `visium_hd/config.py`:

```python
DATA_ROOT = Path("/date/hanyu/JYX/data/visiumHD")

SAMPLES = {
    "Mouse_Kidney_FFPE": {
        "tissue_image": "Visium_HD_Mouse_Kidney_tissue_image.btf",
        "raw_dir":      DATA_ROOT / "Mouse_Kidney",
        "zarr_path":    DATA_ROOT / "Mouse_Kidney" / "data.zarr",
        "raster_zarr":  DATA_ROOT / "Mouse_Kidney" / "data_with_rasters.zarr",
        "out_root":     DATA_ROOT / "Visium_HD_Mouse_Kidney_FFPE",
        "layer":        "20240612",
        "affine": [[7.30294570e+00, -2.03362911e-02, 4.68210838e+03],
                   [-2.04357117e-02, -7.30278299e+00, 2.67365002e+04]],
    },
}

ACTIVE_SAMPLE = "Mouse_Kidney_FFPE"
```

`affine` is the one value that cannot be derived from the data: it is the 2 x 3 registration matrix
that maps rasterised 2 um bin coordinates onto the full-resolution H&E frame, and it has to be
estimated per section. Everything else is path configuration. `layer` is the date sub-folder inside
`WSI/`, `HR_ST/` and `spot_ST/`, matching the convention in the other two branches.

The steps take an optional `--sample` to override `ACTIVE_SAMPLE`, so a second section can be
processed without editing the file.

### 4.4 How tiles are selected

The tile grid is anchored to the **transcript raster**, not to the slide. The four corners of
`rasterized_002um` are projected through the registration matrix, the bounding box of that projection
is grown by `crop_wsi_pixel / 4` and clipped to the slide, and tiles are laid out inside it on the
same 50 % overlapping grid the other branches use.

Two consequences follow, and both matter when matching tiles across branches:

- Tile indices are **relative to that bounding box**, so `0_0` is the top-left corner of the region
  the transcripts cover, not the top-left corner of the slide. Visium and Xenium number tiles from the
  slide origin.
- A slide much larger than its capture area produces tiles only over the capture area, which is what
  keeps a 60 000 px section from generating tens of thousands of empty tiles.

Within a tile, each 320 x 320 sub-patch must pass a tissue filter before it enters `320_to16.npy`:

- its share of near-flat pixels (Sobel magnitude <= 20) must be below `FLAT_RATIO` = 0.75, **and**
- mean luminance < 220, **and** mean red channel > 120.

Accepted sub-patches are resized to 16 x 16, cycled until the stack holds 256 entries, and sorted by
ascending **image entropy** in this historical preprocessing file. That entropy is **not** the
manuscript's density-plus-nuclear-morphology complexity. The current loader reconstructs
the 256 H&E cellular tiles in spatial row-major order and reads a separate
`complexity_grid.npy` scored from registered native H&E and existing nuclear labels.

If no sub-patch passes, the tile still gets `5120.npy` and `5120_to256.npy` but no `320_to16.npy`.
Since the resume check tests for `5120.npy`, such a tile is never revisited; the dataloader, which
needs all four files, will skip it.

### 4.5 How transcripts are binned

`step6_extract_st_grids.py` reads one gene at a time out of `rasterized_002um`, warps that 2 um image
into the tile frame with `cv2.warpAffine` using the registration matrix with the tile origin
subtracted, and block-sums the result at three scales:

| Output | Factor | Grid | Bin |
| --- | --- | --- | --- |
| `HR_ST_256.npz` | 20 | 256 x 256 | 10 um |
| `HR_ST_128.npz` | 40 | 128 x 128 | 20 um |
| `spot_ST.npz` | 200 | 26 x 26 | 100 um |

5120 is not divisible by 200, so the spot grid is padded to 26 x 200 = 5200 before summing; without
that pad the last row and column of transcripts would be dropped. Genes are processed in batches of
`--gene-batch` so that only a slice of the raster is held in memory at a time.

The gene column order is written to `WSI/extract/{layer}/gene_names.csv` and must be kept: the
dataloader indexes the grids by it.

### 4.6 Gene symbols

Visium HD panels contain duplicated gene symbols, but `.sel(c=...)` on the raster requires the
coordinate to be unique. `step4_rasterize_bins.py` therefore renames repeats: the first occurrence
keeps its symbol, later ones become `GENE_1`, `GENE_2`, and so on, in order of appearance. The
renaming happens once, in step 4, and everything downstream reads the already-unique names.

### 4.7 Run order

```bash
cd data_preprocessing/visium_hd
python step1_download.py            # or copy the 10x bundle in yourself
python step2_export_h5ad.py         # optional, for scanpy work
python step3_build_zarr.py
python step4_rasterize_bins.py
python step5_extract_he_tiles.py    # H&E tiles
python step6_extract_st_grids.py    # transcripts -> HR_ST / spot_ST
python step7_stitch_preview.py      # optional visual check
python step8_qc_tile.py --tile 3_3  # optional visual check
```

Steps 5 and 6 are independent of each other and both restartable, so they can be run in either order
or resumed after an interruption. `--workers N` controls the process pool; `--limit N` processes only
the first N tiles, which is the cheap way to check a new registration matrix.

### 4.8 Caveats

- **The registration matrix is per section and is not checked.** Nothing validates that `affine` is
  right; a wrong matrix produces tiles that look plausible but pair the wrong H&E field with the
  wrong transcripts, and only the step 8 overlay or the stitched preview reveals it. Always check a
  tile with `step8_qc_tile.py` on a new sample.
- **`data_with_rasters.zarr` must exist before steps 5 and 6.** Both read the rasterised store, not
  the raw binned outputs; running them against `data.zarr` from step 3 raises a missing-element error.
- **`step6` re-opens the Zarr once per tile.** `SpatialData` objects cannot be shared across
  processes, so each worker reads the store in its own process. This is correct but not cheap on a
  store that has to be re-read from disk; keep `--workers` modest if the store is on a network mount.
- **Only tiles that already exist are binned.** Steps 5 and 6 each recompute the same tile grid
  independently. If the tiling constants or the registration matrix change between the two runs, the
  grids no longer agree and every tile is written twice under different index conventions. Run them
  with the same config, and clear the output tree when you change either.
- **The tile loop stops one step before the region edge.** Tiles are emitted while the tile start is
  at or before `x_max - 5120`, so a region that is not a whole number of tile steps wide loses up to
  one 2560 px strip at its bottom and right edge. For example, a 6000 px region yields a single tile
  covering the first 5120 px. The `crop_wsi_pixel / 4` margin on the projected bounding box absorbs
  part of that strip, but not all of it: if your transcripts reach the edge of the capture area,
  widen the margin in `step5_extract_he_tiles.py` and `step6_extract_st_grids.py`. Changing this
  boundary changes the tile set and requires regenerating the processed data.
- The historical image-entropy sort, 320 x 320 sub-patch filter and 50 % overlap
  occur in the Visium and Xenium branches. Their entropy sort does not determine
  the current model's complexity selection; `complexity_grid.npy` does.
- `WSI_tile.png` is written for every tile. It is a convenience preview and roughly doubles the I/O
  of step 5; drop that write for a large cohort.

---

## 5. Xenium (`xenium/Xenium_pami.py`)

For a bounded, parameterized extraction from the original 10x mouse-pup files, use
`xenium/replay_registered_tile.py`. It reads `experiment.xenium`, `gene_panel.json` and
`he_imagealignment.csv` from the same raw sample directory. The two large inputs can
be local files or matching HTTP range URLs:

```bash
RAW=/path/to/Xenium_V1_mouse_pup/outs
BASE=https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_mouse_pup
python data_preprocessing/xenium/replay_registered_tile.py \
  --raw-dir "$RAW" \
  --he "$BASE/Xenium_V1_mouse_pup_he_image.ome.tif" \
  --transcripts "$BASE/Xenium_V1_mouse_pup_transcripts.parquet" \
  --tile 3_10 --output /path/to/replayed/3_10
```

The command writes `5120.npy`, `5120_to256.npy`, `HR_ST_256.npz`,
`HR_ST_128.npz`, `spot_ST.npz`, `gene_names.csv` and `provenance.json`.
The three count matrices retain every panel channel in the original `gene_panel.json`
order and have exactly equal per-gene totals. Coordinates are in Xenium microns;
tile `3_10` starts at `(3840, 12800)` µm. It includes all panel transcripts in the
field without a QV or cell-assignment filter. Only intersecting Parquet row groups
and source H&E TIFF tiles are read, so unrelated tissue fields are not decoded.
The H&E resampler is OpenCV bilinear; a previously processed tree made with
libvips bilinear can differ by a few uint8 levels while retaining the same
registration. Check the native image against its recorded transform and verify
the counts before supplying any other conditions or calling the formal loader.

The notebook-style `Xenium_pami.py` below describes the earlier full-slide
pipeline. It has sample-specific path constants and is not the command above.

### 5.1 What it does

```
morphology.ome.tif ──┐
alignment.csv ───────┼─► affine-warped, cropped, 0.5 um/px H&E ─► WSI tiles (5120_to256, 320_to16, 5120)
aligned_HE.tif ──────┘
transcripts.parquet ─┬─► HR_ST_256.npz  (256x256, 10 um/bin)
gene_groups.csv ─────┴─► HR_ST_128.npz  (128x128, 20 um/bin)
                         spot_ST.npz    (26x26,   100 um/bin)
```

### 5.2 Inputs

| File | Used for |
| --- | --- |
| `morphology_img_info.npy` | `[height, width, pixel_size_um]` of the morphology image, cached by the commented-out block at the top of the script |
| `morphology.ome.tif` / `TMA1_40X_ZLIB_2_1024.ome.tif` | the raw morphology image, read with `pyometiff` |
| `alignment.csv` | a 2 x 3 affine matrix mapping morphology pixels onto the registered H&E frame |
| `aligned_HE.tif` | the already-registered H&E image; if present it replaces the warped array (see below) |
| `transcripts.parquet` | per-transcript `x_location`, `y_location` (um) and gene symbol |
| `gene_groups.csv` | the gene panel, produced by `jsontocsv.py` |

`morphology_img_info.npy` is a small cache written by the commented block near the top of
`Xenium_pami.py`. Uncomment it for the first run on a new sample to avoid re-reading the OME-TIFF
metadata on every run.

The alignment matrix is applied only when the H&E still needs registering:

```python
if img_array.shape[0] < 32766 and img_array.shape[1] < 32766:
    img_array = cv2.warpAffine(img_array, alignment, (W, H))
else:
    img_array = warpAffine_new(img_array, alignment, (W, H))   # pure-NumPy bilinear warp
    # then crop the black border introduced by the warp
```

`warpAffine_new` is a pure-NumPy reimplementation of bilinear `cv2.warpAffine`, selected above the
`32766` px guard; it is functionally equivalent but slower and much more memory-hungry, since it
materialises full-image `float64` coordinate grids. `utils.py` also defines a tiled variant,
`warpAffine_new1`, which warps the image in `tile_size` blocks and warps each tile with OpenCV —
prefer it if `warpAffine_new` exhausts memory on a large slide.

**Note the last two lines of the WSI block**: if `aligned_HE.tif` exists next to the sample, `im` is
overwritten with it (`im = plt.imread(root + 'aligned_HE.tif', format="tiff")`) and the warping work
above is discarded. That is the intended path when the H&E has already been registered externally —
just be aware that the alignment block is then dead code.

### 5.3 Gene panel

`jsontocsv.py` flattens the Xenium `gene_panel.json` into `gene_groups.csv`:

```python
with open('.../gene_panel.json') as f:
    data = json.load(f)
names = [{'name': t['type']['data']['name']}
         for t in data['payload']['targets']
         if 'type' in t and 'data' in t['type'] and 'name' in t['type']['data']]
```

Edit the two hard-coded paths at the top and run it once per panel:

```bash
cd data_preprocessing/xenium
python jsontocsv.py
```

The resulting order defines the column order of every `HR_ST_*.npz` / `spot_ST.npz`, so **do not
regenerate or reorder this file after you have produced ST grids** — the bins would no longer line up
with the gene indices.

### 5.4 Configuration

The variables are declared in **two** blocks — one above the H&E pass and one above the ST pass — and
several are re-assigned in the second block, so the later value wins:

```python
# block 1, above the H&E pass
root            = r'/home/cbtil4/ST/datasets/output-XETG00166__0041204__TMA1__20241031__051835/'
save_HE_root    = r'/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/WSI/extract/20241031/'
ori_HE_name     = 'TMA1_40X_ZLIB_2_1024.ome.tif'
alignment_file  = 'alignment.csv'

root            = r'/home/cbtil4/ST/datasets/'                              # overrides the line above
save_HE_root    = r'/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/'   # overrides the line above

# block 2, above the ST pass
gene_groups     = r'/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/gene_groups.csv'
root            = r'/home/cbtil4/ST/datasets/output-XETG00166__0041204__TMA1__20241031__051835/'
save_HRST_root  = r'.../Xenium5k_Craniopharyngioma/HR_ST/extract/20241031/'
save_spotST_root= r'.../Xenium5k_Craniopharyngioma/spot_ST/extract/20241031/'
save_WSI_root   = r'.../Xenium5k_Craniopharyngioma/WSI/extract/20241031/'
```

When switching samples, update the sample folder name
(`output-XETG00166__0041204__TMA1__20241031__051835`), which is hard-coded in the `morphology_img_info.npy`
load as well as in both `root` assignments.

### 5.5 Run order

`Xenium_pami.py` is written as a linear notebook-style script with two independent passes, and the two
passes do not talk to each other: **run the H&E pass first**, then the ST pass.

The ST pass loops over the *whole* image grid and creates `HR_ST/`/`spot_ST/` tile folders
unconditionally, so it will happily write ST grids for tiles the H&E pass rejected (blank or
over-exposed regions). Those tiles then have no matching `WSI/` patch, which breaks the
patch/expression pairing the dataloader relies on. The commented-out guard in `main()` is meant to
prevent exactly that:

```python
tile_dir_wsi = os.path.join(save_WSI_root, f"{i}_{j}")
if not os.path.isdir(tile_dir_wsi):
    continue
```

Note that as written this guard would never fire: the H&E pass also calls `os.makedirs` for every
`{i}_{j}` before it filters sub-patches, so the directory exists whether or not any patch was kept.
Check for the file the H&E pass actually writes, and skip tiles that lack it:

```python
if not os.path.exists(os.path.join(save_WSI_root, f"{i}_{j}", "5120_to256.npy")):
    continue
```

Note also that the two passes share the global `root`, which block 2 re-assigns to the sample folder
(see 4.4), and that `crop_wsi_mpp` / `st_rela_wsi_mpp` / `spotst_rela_wsi_mpp` / `gene_groups` are only
defined in block 2 — so `main()` cannot be imported and run on its own without that block executing
first.

### 5.6 Run

1. **H&E / WSI pass** — the top-level `for i / for j` loop (writes `320_to16.npy`, `5120.npy`,
   `5120_to256.npy`). Run the file as-is.
2. **HR_ST / spot_ST pass** — the `main()` at the bottom of the file, which reads
   `transcripts.parquet` through DuckDB and writes the three ST grids. Call it from the end of the
   script (`main()`) or paste the block into a second file.

```bash
cd data_preprocessing/xenium
python Xenium_pami.py          # step 1: H&E tiles; add `main()` at the end for step 2
```

### 5.7 How transcripts are binned

`main()` runs one SQL query per tile:

```sql
WITH t AS (
  SELECT x_location, y_location, CAST(gene_symbol AS VARCHAR) AS gene_symbol
  FROM read_parquet('transcripts.parquet')
  WHERE x_location >= x0 AND x_location < x1
    AND y_location >= y0 AND y_location < y1
)
SELECT x_location, y_location, t.gene_symbol FROM t JOIN genes_df g USING (gene_symbol)
```

The join against the registered gene panel is what restricts the counts to the panel. Coordinates are
then converted to bin indices per scale, clipped to the grid, and accumulated with
`scipy.sparse.coo_matrix(...).tocsr()`, which sums duplicate `(bin, gene)` entries. Column names are
auto-detected (`x_location` / `y_location` / one of `gene_symbol`, `gene`, `feature_name`, `geneid`,
`symbol`), so a panel rename in a newer Xenium release is usually handled without editing the script.

### 5.8 Caveats

- **Blank tiles are skipped rather than crashing.** When no 320 x 320 sub-patch passes the
  gradient/brightness filter the tile has no sub-patch stack, and the H&E pass leaves it unwritten
  instead of dereferencing an unassigned array. Such a tile then has no `WSI/` patch, which is the
  correct outcome: there is nothing to pair with an ST grid.
- `remove_all_file` inside `Xenium_pami.py` recurses into nested directories with `shutil.rmtree`; the
  module imports `shutil` for that.
- `utils.py` here defines `warpAffine_new` **and** `warpAffine_new1` (the tiled version) plus several
  debug `print` calls inside `warpAffine_new`. The prints are harmless but very chatty on large images.
- `spotst_rela_wsi_mpp = 200` gives 100 um bins and `st_rela_wsi_mpp = 20` gives 10 um bins; changing
  either changes the grid side (`5120 / value (+ 1)`), which the dataloader assumes is 256 and 26.
- Transcript counts are stored as `int16`; 10 um bins are far below the limit, but a much coarser grid
  on a dense sample could saturate.

---

## 6. Verifying the output

A quick sanity check that the layout and shapes are right:

```python
import scipy.sparse as sp, numpy as np, os

tile = "HR_ST/extract/20241031/0_0"
hr = sp.load_npz(os.path.join(tile, "HR_ST_256.npz"))
print(hr.shape)                       # (65536, G)
print(hr.toarray().reshape(256, 256, -1).sum())   # total transcripts in the tile

he = np.load("WSI/extract/20241031/0_0/5120_to256.npy")
print(he.shape, he.dtype)             # (256, 256, 3) uint8
sub = np.load("WSI/extract/20241031/0_0/320_to16.npy")
print(sub.shape)                      # (256, 16, 16, 3) uint8
```

For every tile, the gene dimension `G` must equal the number of rows in `gene_groups.csv` (Visium,
Xenium) or `gene_names.csv` (Visium HD), and `HR_ST_256.npz`, `HR_ST_128.npz` and `spot_ST.npz` must
all share it. For the 2,560 µm fields and half-open coordinate intervals used here, the three
grids bin the *same* transcripts and their per-gene totals should agree exactly. Because tiles overlap
by 50 %, a transcript near a tile border is counted in up to four neighbouring tiles; that is expected,
and the dataloader treats tiles as separate records. Assign their donor to a split before
patch extraction so overlapping fields cannot cross train/test boundaries.

---

## 7. Additional model inputs

These files also appear in the FOCUS data layout. The table identifies their
source; a complete public mouse-pup example and exact
source entrances are in [real data preparation](../docs/real_data.md).

| File | Produced by |
| --- | --- |
| `cell_mask.npy` | `histology/cell_mask.py` segments H&E with StarDist; the bundled demo instead uses registered 10x polygons packed by `../demos/select_official_cells.py`. |
| `st_resolution.json` | Per-sample measured pre-degradation ST bin width (`original_st_bin_um`) and a physical `source` description. The model uses this value to determine whether the original ST scale exceeds 10 µm; tensor size is not a substitute. |
| `pre_he.npy` | Official [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) tile features, 257 vectors of 1536 dimensions; export with `export_conditioning.py` after accepting the model access terms. |
| `complexity_grid.npy` | Density and native nuclear size, shape and hematoxylin texture scored from registered H&E plus existing nuclear-instance labels with `score_complexity.py`; this does not generate masks. The scorer writes a source-hash `.json` sidecar. |
| `scgpt_data/` | Official [scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial) embeddings per five-gene group from `export_conditioning.py`; see the pinned non-Flash export and ortholog mapping in `real_data.md`. |
| `sc/{scRNA_data}.h5ad` | Independent tissue/stage matched raw-count reference; `extract_mouse_pup_reference.py` selects the public P0 muscle subset for the demo. |
| `coexpression_matrix.npy` | Tissue spatial gene-gene prior with exact row-name sidecar. See the [sample-level Pearson and tissue-mean example](../docs/real_data.md#preparing-a-tissue-co-expression-prior). The bundled demo uses its existing same-tile matrix. |
| `gene_order.npy`, `gene_name.txt` | the gene indices/names the model is trained on, derived from `gene_groups.csv` |

Cell-type annotation, deconvolution and trajectory inference are also outside this folder and are
documented in [`../downstream/README.md`](../downstream/README.md).

The authors' external data link and the complete loader input format are described in
[the main README](../README.md#data-preparation).
