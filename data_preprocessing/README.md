# Data Preprocessing

This folder contains the code that converts raw **10x Visium / VisiumHD** and **10x Xenium**
output into the per-patch layout that FOCUS trains and evaluates on. It covers the
**H&E tiling** and the **transcript-to-grid binning** steps; the embedding, segmentation and
co-expression steps that also feed the model are listed at the [end of this document](#6-what-is-not-covered-here).

```
data_preprocessing/
├── visium/
│   ├── visium.py        # Visium / VisiumHD: H&E tiles + spot_ST grid
│   └── utils.py         # h5 matrix reader, gradient magnitude, graph helpers
└── xenium/
    ├── Xenium_pami.py   # Xenium: aligned H&E tiles + HR_ST and spot_ST grids
    ├── utils.py         # OME-TIFF affine warping, gradient magnitude, graph helpers
    └── jsontocsv.py     # Xenium gene_panel.json -> gene_groups.csv
```

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
│   │   └── HR_ST_128.npz        # (16384, G) sparse, 128x128 grid at 20 um/bin   (Xenium only)
│   ├── spot_ST/extract/{layer}/{i}_{j}/
│   │   └── spot_ST.npz          # (676, G) sparse, 26x26 grid at 100 um/bin
│   ├── WSI/extract/{layer}/{i}_{j}/
│   │   ├── 5120_to256.npy       # (256, 256, 3) uint8, the tile at 10 um/px
│   │   ├── 320_to16.npy         # (256, 16, 16, 3) uint8, sub-patches at 10 um/px
│   │   └── 5120.npy             # (5120, 5120, 3) uint8, full-resolution tile (Xenium only)
│   ├── gene_order.npy           # produced downstream, see section 6
│   └── gene_name.txt            # produced downstream, see section 6
└── coexpression_matrix.npy      # produced downstream, see section 6
```

`gene_groups.csv` sits at the cohort level because the gene panel is shared by every sample of a
tissue: the Visium script reads `super_root/gene_groups.csv`, the Xenium script the path in its
`gene_groups` variable.

### Why those numbers

The whole layout follows from the constants both platforms share:

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
a Visium-like 100 um spot pitch, which is what makes the Xenium and Visium branches interchangeable
for the dataloader.

`HR_ST_128.npz` is the same transcripts re-binned to 128 x 128 (20 um/bin) and is only written by the
Xenium branch.

### Sparse matrix convention

`HR_ST_*.npz` and `spot_ST.npz` are `scipy.sparse` CSR matrices saved with `scipy.sparse.save_npz`.
The 2-D grid is flattened row-major, so element `(h, w, g)` of the grid is row `h * W + w` and column
`g` of the matrix, where `g` indexes `gene_groups.csv`. Load them with:

```python
import scipy.sparse as sp
m = sp.load_npz("HR_ST_256.npz")            # (65536, G)
grid = m.toarray().reshape(256, 256, -1)    # (h, w, gene)
```

Counts are accumulated in `int16`, so a single bin saturating above 32767 transcripts would overflow;
this does not happen at 10 um bins, but keep it in mind if you re-bin to a much coarser grid.

---

## 2. Requirements

```bash
pip install numpy pandas opencv-python Pillow h5py scipy scikit-image tqdm duckdb pyometiff matplotlib scikit-learn anndata torch
```

Notes on the less obvious ones:

- **`duckdb`** — Xenium only. `Xenium_pami.py` pushes the per-tile transcript window into DuckDB and
  joins it against the gene panel in SQL, which is what keeps the parquet scan from loading the whole
  `transcripts.parquet` into memory.
- **`pyometiff`** — Xenium only, for reading the `.ome.tif` morphology image.
- **`torch` / `anndata` / `scikit-learn` / `matplotlib`** are imported by both `utils.py` files. The
  functions that need them (graph construction, LSI, CLR normalisation) are not used by the
  preprocessing entry points, but the imports are unconditional — install them or trim the imports.
- **`h5py`** is used by `get_matrix_from_h5` in both `utils.py` files; the original `pytables`
  implementation is kept commented out just above it.
- `PIL.Image.MAX_IMAGE_PIXELS` and `OPENCV_IO_MAX_IMAGE_PIXELS` are raised in `utils.py`; whole-slide
  images are far larger than Pillow's default decompression-bomb limit.

Both scripts are plain Python entry points driven by constants at the top of the file rather than by
CLI arguments, so there is nothing to install beyond the dependencies and nothing to pass on the
command line — edit the paths, then run the file.

---

## 3. Visium / VisiumHD (`visium/visium.py`)

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

- **`import json2`** at the top of `visium.py` is not a PyPI package; it is a local helper that must
  expose `load_file(path)`. If you do not have it, either drop a `json2.py` next to the script:

  ```python
  import json
  def load_file(path):
      with open(path) as fh:
          return json.load(fh)
  ```

  or replace the two uses with `json.load(open(...))`.
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

## 4. Xenium (`xenium/Xenium_pami.py`)

### 4.1 What it does

```
morphology.ome.tif ──┐
alignment.csv ───────┼─► affine-warped, cropped, 0.5 um/px H&E ─► WSI tiles (5120_to256, 320_to16, 5120)
aligned_HE.tif ──────┘
transcripts.parquet ─┬─► HR_ST_256.npz  (256x256, 10 um/bin)
gene_groups.csv ─────┴─► HR_ST_128.npz  (128x128, 20 um/bin)
                         spot_ST.npz    (26x26,   100 um/bin)
```

### 4.2 Inputs

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

### 4.3 Gene panel

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

### 4.4 Configuration

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

### 4.5 Run order

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

### 4.6 Run

1. **H&E / WSI pass** — the top-level `for i / for j` loop (writes `320_to16.npy`, `5120.npy`,
   `5120_to256.npy`). Run the file as-is.
2. **HR_ST / spot_ST pass** — the `main()` at the bottom of the file, which reads
   `transcripts.parquet` through DuckDB and writes the three ST grids. Call it from the end of the
   script (`main()`) or paste the block into a second file.

```bash
cd data_preprocessing/xenium
python Xenium_pami.py          # step 1: H&E tiles; add `main()` at the end for step 2
```

### 4.7 How transcripts are binned

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

### 4.8 Caveats

- **Blank tiles crash the H&E pass.** If no 320 x 320 sub-patch passes the gradient/brightness filter,
  `rgba_image_320_save` is never assigned and the `WSI_320.shape[0]` line raises `UnboundLocalError`.
  Guard the block with `if count == 0: continue` when processing slides with large empty regions.
- **`shutil` is used but not imported** in the `remove_all_file` defined inside `Xenium_pami.py`
  (it only triggers when a nested directory is found). Add `import shutil` at the top, or use the
  `remove_all_file` from `utils.py`, which has no such branch.
- `utils.py` here defines `warpAffine_new` **and** `warpAffine_new1` (the tiled version) plus several
  debug `print` calls inside `warpAffine_new`. The prints are harmless but very chatty on large images.
- `spotst_rela_wsi_mpp = 200` gives 100 um bins and `st_rela_wsi_mpp = 20` gives 10 um bins; changing
  either changes the grid side (`5120 / value (+ 1)`), which the dataloader assumes is 256 and 26.
- Transcript counts are stored as `int16`; 10 um bins are far below the limit, but a much coarser grid
  on a dense sample could saturate.

---

## 5. Verifying the output

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

For every tile, the gene dimension `G` must equal the number of rows in `gene_groups.csv`, and
`HR_ST_256.npz`, `HR_ST_128.npz` and `spot_ST.npz` must all share it. The three grids bin the *same*
transcripts at different scales, so their totals should agree to within the few transcripts that land
exactly on a window edge — a useful check that the binning scales are consistent. Because tiles overlap
by 50 %, a transcript near a tile border is counted in up to four neighbouring tiles; that is expected,
and the dataloader treats tiles as independent samples.

---

## 6. What is not covered here

These files appear in the FOCUS data layout but are produced by separate steps, not by the two scripts
in this folder:

| File | Produced by |
| --- | --- |
| `cell_mask.npy` | cell segmentation on the H&E tile (e.g. CellViT / StarDist) |
| `pre_he.npy` | [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath) H&E features, 512-dim |
| `scgpt_data/` | [scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial) embeddings per gene group |
| `sc/{scRNA_data}.h5ad` | the matched single-cell reference for the tissue |
| `coexpression_matrix.npy` | spatial gene-gene co-expression computed across the cohort |
| `gene_order.npy`, `gene_name.txt` | the gene indices/names the model is trained on, derived from `gene_groups.csv` |

If you would rather not run any of this, the already-processed cross-platform multimodal dataset is
available for download — see the
[Curated Cross-platform Multimodal Dataset](../README.md#curated-cross-platform-multimodal-dataset)
section of the main README.
