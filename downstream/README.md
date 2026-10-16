# Downstream Analysis

This folder holds the analyses that run on the spatial sample after the FOCUS
patch layout exists: cell-level annotation of one section, reference-based
deconvolution, aggregation of Visium HD 2 um bins into cells, and pseudotime
trajectory inference.

The scripts read processed spatial data and references independently of the training code.
Their outputs are listed in section 7.

```
downstream/
├── deconvolution/     RCTD reference -> spatial deconvolution -> QC   (baseline)
├── annotation/        method 7 annotation, plus a Tangram baseline
├── matrix/            Visium HD bin -> cell
└── trajectory/        preprocessing, then PSTS trajectories
```

## What each annotation route is for

Two different things live in `annotation/` and `deconvolution/`, and they are not
two attempts at the same step. The method section describes one annotation
procedure; the other two are comparators it is benchmarked against.

| Route | What it is | Where |
| --- | --- | --- |
| Scanpy clustering + Wilcoxon markers + manual annotation | **the annotation used for the results.** Highly variable genes, PCA, a kNN graph, UMAP and Leiden clustering, then `rank_genes_groups(method="wilcoxon")`; the cluster labels are assigned by hand from the ranked markers | [`annotation/scanpy_annotation.py`](annotation/scanpy_annotation.py) |
| RCTD (Cable et al., 2022) | a **baseline comparator**, reproduced here. It deconvolves each spot against a scRNA-seq reference instead of clustering the section | [`deconvolution/`](deconvolution/) |
| Tangram (Biancalani et al., 2021) | a **baseline comparator**, reproduced here. It maps a reference onto the section and is used as the sensitivity comparison | [`annotation/tangram_reference.py`](annotation/tangram_reference.py) |
| `coarse_annotation.py` | builds the label vector for a section from the RCTD and geo-labelled sources. Part of the baseline route, not of the method | [`annotation/coarse_annotation.py`](annotation/coarse_annotation.py) |

So the method route and the baseline route both exist and both run; they answer
different questions. Nothing in `deconvolution/` feeds the annotation that the
paper reports.

---

## 1. Requirements

```bash
python -m pip install ".[downstream]"
```

Run the command from the repository root. Version constraints are listed in `setup.cfg`.
The extra includes OpenSlide and its native library, PyArrow for Space Ranger parquet files,
and igraph/Leiden dependencies for clustering.

The deconvolution and trajectory steps were run with a dedicated environment:

```
/data/compute_qnf/miniforge3/envs/ccc_spatial/bin/python   # rctd-py 0.3.8
```

`bin2cell.py` additionally needs `shapely` (the STRtree spatial index) and
`opencv-python` for the marker maps. `02_psts.py` reimplements the stLearn
pipeline directly and therefore does **not** need `stlearn` installed.

Every script is a plain entry point driven by constants at the top of the file;
fill in the paths for your project and run the file.

---

## 2. Sample and inputs

The scripts are written for one section of the bone cohort, **16-4-NC**
(184,589 segmented cells x 19,059 genes). The paths below are the ones used for
the manuscript; change them in the `BASE` block at the top of each script.

| Input | Path |
| --- | --- |
| Cell-level spatial object | `xiao/16-4-NC_SR.h5ad` |
| Visium HD bins (optional alternative) | `ST/AXB-1-2-A1/16-4-NC_SR/binned_outputs/square_008um` |
| scRNA-seq references | `xiao/young_scrna_anno.h5ad`, `xiao/YBM_anno_add_adipoMSC.h5ad` |
| Segmentation polygons | `cell_segmentation/*_cell_data.csv` |
| Independent RCTD result (QC only) | `xzq/.../soj_rctd_first_type_nc_min20.csv` |

Input conditions:

- The spatial data is **nuclear** segmentation. Cell types that have no nucleus
  (erythrocytes, platelets) cannot be detected in it, so they are removed from
  the reference rather than being left to absorb probability mass.
- The reference atlases carry different normalisation states: one is stored
  log-normalised and is inverted with `expm1` (lossy), the other holds raw
  counts. `02_run_deconvolution.py` prints a warning when it has to do this.

---

## 3. Deconvolution (`deconvolution/`) — baseline

A standard three-step RCTD workflow, reproduced as a **comparator**, not as the
annotation the results use. rctd-py in `doublet` mode builds one joint
reference from the scRNA-seq atlases and fits it to the spatial data.

| Step | Purpose |
| --- | --- |
| `01_build_reference.py` | merge the two atlases, map fine labels to the 13-class coarse scheme, filter enucleated cells and fine types with <= 25 cells; writes `Reference_MetaData_cleaned.csv.gz` and `bone_labels.csv` |
| `02_run_deconvolution.py` | fit the reference to the spatial sample; writes the four-column result table and an AnnData object with fine and coarse proportion matrices |
| `03_qc.py` | composition, canonical-marker detection, and agreement with an independent RCTD run |

Run them in order:

```bash
cd downstream/deconvolution
python 01_build_reference.py
python 02_run_deconvolution.py
python 03_qc.py
```

`02_run_deconvolution.py` accepts either a cell-level `.h5ad` or a Space Ranger
bin directory as the spatial input; both are handled by the same code path.
Samples whose outputs already exist are skipped, so an interrupted run can be
restarted. For long runs, start the script in the background:

```bash
setsid nohup python 02_run_deconvolution.py > output/run_02.log 2>&1 &
```

### Notes

- The reference lacks osteocytes and adipocytes, so RCTD cannot assign these labels.
- The coarse label set is the one the manuscript figures use. Anything you
  change in `FINE_TO_COARSE` must be changed in `03_qc.py` as well, or the
  agreement check compares two different taxonomies.

---

## 4. Annotation (`annotation/`)

Two routes sit here. `scanpy_annotation.py` is the annotation procedure the
results use; the other two scripts belong to the RCTD/Tangram baseline route.

### 4.1 The annotation procedure — `scanpy_annotation.py`

The section is clustered and markers are ranked for manual assignment: highly
variable genes, PCA, a kNN graph, UMAP and Leiden clustering, then
`rank_genes_groups(method="wilcoxon")`. The output is the ranked marker table per
cluster, plus the clustered object with the Leiden labels attached; the labels
themselves are assigned by hand from those markers against canonical lineage and
state markers.

```bash
cd downstream/annotation
python scanpy_annotation.py --input <section>.h5ad --name 16-4-NC
```

### 4.2 The baseline route — `coarse_annotation.py`, `tangram_reference.py`

Two label sources for the same section, used to build the RCTD/Tangram
comparison.

`coarse_annotation.py` builds a label vector from the baseline sources: the
YBM-reference RCTD result is taken first, the geo-labelled annotation fills the
cells it did not cover, and whatever remains stays `Unassigned`. It writes the
annotated object, the label table and the spatial map.

`tangram_reference.py` addresses the two classes the YBM reference cannot
resolve. A T2DM scRNA-seq reference is mapped onto the section with Tangram, but
the prediction is only accepted for classes the YBM reference lacks, and only
above a confidence threshold — for everything else the existing label wins.

```bash
cd downstream/annotation
python tangram_reference.py
python coarse_annotation.py
```

Note the ordering: `coarse_annotation.py` reads the Tangram output, so it must be
run second if you want the supplemented labels.

Both scripts are secondary to `deconvolution/`. The YBM-reference workflow does
its own deconvolution and is the source to prefer; these two exist to cover the
label sources that predate it.

---

## 5. Bin-to-cell aggregation (`matrix/bin2cell.py`)

Visium HD 2 um bins are aggregated into cells: each bin is tested against the
segmented nuclei with an STRtree point-in-polygon query, and the counts of the
bins inside a polygon are summed. Bins that miss every polygon are dropped, as
are cells that receive no bin.

Coordinate convention is `X = pxl_col_in_fullres`, `Y = pxl_row_in_fullres`,
matching the rest of the project.

```bash
cd downstream/matrix
python bin2cell.py
```

The output was checked against the segmentation tables:
`n_obs` equals the row count of the corresponding `cell_segmentation/*_cell_data.csv`
for every sample.

---

## 6. Trajectory inference (`trajectory/`)

`01_preprocess.py` filters the section, normalises, clusters, and scores the bone
lineage markers. `02_psts.py` then runs the stLearn PSTS trajectory.

```bash
cd downstream/trajectory
python 01_preprocess.py
python 02_psts.py
```

`02_psts.py` reimplements the stLearn pipeline rather than calling it:
DBSCAN localisation of subclusters, PAGA, diffusion pseudotime, and the global
pseudotime space, followed by a minimum spanning arborescence over the subcluster
graph whose edges are weighted by both expression distance and spatial distance.
That spatial term is what keeps the pseudotime consistent with the tissue layout
instead of following the transcriptome alone. One trajectory is produced per
lineage (osteogenic, adipogenic, chondrogenic).

Outputs and required intermediate data:

- The step computes the trajectories and stops. Figure assembly is separate;
  `trajectories` holds the trees, subcluster centroids and ordered node paths.
- `02_psts.py` expects `nc_psts_trajectory.h5ad`, the object produced by
  subsampling the scored section to 50,000 cells and assigning k-means clusters.
  That subsampling script is not in this folder.

---

## 7. Output layout

Each script writes next to itself, under `output/`:

```
downstream/deconvolution/output/
├── Reference_MetaData_cleaned.csv.gz
├── bone_labels.csv
├── DeconvolutionResults_16-4-NC.csv.gz
├── 16-4-NC_deconvolved.h5ad
├── qc_report.txt
├── qc_per_sample.csv
├── qc_markers_16-4-NC.csv
└── qc_vs_xzq_16-4-NC.csv

downstream/annotation/output/
├── 16-4-NC_SR_coarse_annotated.h5ad
├── coarse_celltype_nc.csv
└── coarse_celltype_nc.{png,pdf}

downstream/matrix/output/
├── h5ad/<sample>.h5ad
└── figures/<sample>/*.png

downstream/trajectory/output/
├── nc_preprocessed.h5ad
└── figures/
```
