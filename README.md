# Coordinated Enhancement of Molecular and Spatial Fidelity for Biological Inference in Spatial Transcriptomics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Open-source license:** FOCUS source code is released under the [MIT License](LICENSE).

## Overview

Spatial transcriptomics (ST) measures gene expression within tissue, but its capture units impose a trade-off between resolution and sensitivity. Smaller units locate transcripts more precisely but capture fewer molecules. Larger units combine signals from neighbouring cells, blurring tissue boundaries. Correcting these effects separately can leave expression profiles inconsistent with the tissue image or the cells they represent.

FOCUS is a multimodal generative foundation model built on conditional diffusion. It combines native ST with paired haematoxylin and eosin (H&E) images, cell masks, single-cell RNA-sequencing references and spatial co-expression priors. Three task-specific modules address sensitivity, alignment with tissue morphology and cell-level fidelity. A cross-task coordinator compresses active modules into tokens, exchanges information through multi-head cross-attention and uses feature-wise linear modulation (FiLM) to condition each module on the others.

The study trained and evaluated FOCUS across ten ST platforms, 34 tissue contexts and over 1.7 million tissue patches, with more than 5.8 million single-cell reference profiles. Joint enhancement improved performance over treating the tasks separately. Applications examined neutrophil–tumour interactions, spatial osteocyte differentiation and primary–metastatic programmes in head and neck squamous cell carcinoma (HNSCC).

![Figure 1 | Coupled ST limitations and FOCUS overview](FOCUS.png)

**Figure 1 | Coupled ST limitations and FOCUS overview.** (**a**) Illustration of spatial gene profiling of heterogeneous tissue. (**b**) Three representative coupled limitations: resolution-sensitivity, resolution-alignment, and sensitivity-cell-level fidelity in cellularly dense tissues. (**c**) FOCUS integrates native ST, paired H&E with cell masks, scRNA-seq references and spatial co-expression priors through task-specific modules, coordinated by a cross-task strategy for joint enhancement of resolution, sensitivity, molecular-tissue alignment and cell-level fidelity.

## Repository Structure

```
FOCUS/
├── train.py                     # training entry point
├── train_part.py                # gene-group loop around the training run
├── sample.py                    # inference / evaluation entry point
├── FOCUS.yaml                   # conda environment
├── requirements.txt             # core pip dependencies
├── setup.py / setup.cfg         # package metadata and build configuration
├── pyproject.toml / MANIFEST.in # build backend and source-package contents
├── CONTRIBUTING.md / CHANGELOG.md
├── Dockerfile                   # CUDA container recipe
├── demos/                       # input/, run.py, focus-5gene.pt, generated output/
├── docs/                        # usage, algorithm outline and tested environments
├── hooks/ / install-hooks.sh    # optional local whitespace check
├── FOCUS.png / demo_results.png # method overview and archived earlier demo image
├── config/
│   ├── config_train.yaml        # training hyperparameters
│   └── config_test.yaml         # sampling hyperparameters
├── guided_diffusion/            # the model and the diffusion machinery
│   ├── unet.py                  # diffusion U-Net + the task-specific modules
│   ├── gaussian_diffusion.py    # forward/reverse diffusion and the training objective
│   ├── respace.py               # optional explicit shorter sampling schedules
│   ├── sampler.py               # DPM-Solver utilities
│   ├── load_dataset.py          # multi-class patch dataset + dataloader
│   ├── script_util.py           # model/diffusion factory
│   ├── train_util.py            # TrainLoop
│   └── ...
├── data_preprocessing/          # raw Visium / Visium HD / Xenium -> FOCUS patch layout
│   ├── README.md                # step-by-step preprocessing walkthrough
│   ├── visium/
│   ├── visium_hd/
│   └── xenium/
├── downstream/                  # annotation, deconvolution and trajectory analysis
│   ├── README.md                # what each analysis does and how to run it
│   ├── deconvolution/
│   ├── annotation/
│   ├── matrix/
│   └── trajectory/
├── basic_net/                   # UNet building blocks
├── transform/                   # augmentation / transform helpers
└── assets/                      # figures used by this README
```

### Module Map

The live method chain is `UNetModel._forward_manuscript`, with task modules in
`guided_diffusion/manuscript.py` and cell measurements in `guided_diffusion/morphology.py`.

| Paper | Code |
| --- | --- |
| Reference-guided sensitivity | `reference_projection`, `SpatialGeneMemoryBank`, `HEGuidedRegionalAttention` |
| Morphology-guided alignment | `MorphologyAlignment`: local modulation, deformation, NGF, GMC, GMSC |
| Complexity-aware cell fidelity | `complexity_from_native_labels`, `cellular_fusion`, `pearson_cells`, `cell_consistency` |
| Three-task activation and coordination | `task_scores`, `ActiveTaskCoordinator` |
| Pretrained feature matching | `ManuscriptConditioning.he_match` and `st_match` |
| Persistent metadata conditioning | `UNetModel.meta_processor` |
| Freeze-update fine-tuning | `configure_finetuning`, `LowRankUpdate`, prompt-conditioned routing |

The equation mapping and explicit choices are listed in [reproduction.md](docs/reproduction.md).

## Installation

### Environment Setup

The source package is version **0.3.2**. With a supported Python environment and a CUDA-compatible
PyTorch installation, install core dependencies and the package from this checkout:

```bash
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126 --timeout 120
python -m pip install -r requirements.txt --timeout 120
python -m pip install --no-deps .
```

`requirements.txt` lists dependency version ranges. The versions used in our tests are listed below.
The package has not been uploaded to PyPI. Build source and wheel archives with `python -m build`
after installing the `dev` extra. The source archive contains the full scripts, docs and demo data;
the wheel installs the core modules and the demo/config assets under `share/focus/`.

For the preprocessing and downstream scripts, install the additional dependencies:

```bash
python -m pip install ".[preprocessing,downstream]" --timeout 120
```

An alternative Linux conda environment is also supplied:

```bash
conda env create -f FOCUS.yaml
conda activate focus
```

### Requirements

- Python ≥ 3.10 (the environment pins 3.10.19)
- PyTorch 2.9.0 with CUDA 12.6
- Key dependencies:
  - `scanpy` (1.11.5) — ST data processing
  - `anndata` (0.11.4) — data structures
  - `pot` (0.9.6) — optimal transport
  - `transformers` (4.57.1) — scGPT / metadata BERT integration
  - `timm` (1.0.22) — vision models
  - `mpi4py` (3.1.3) — MPI runtime used by the training and sampling entrypoints

`FOCUS.yaml` records a supplied conda environment; it is not a verified record of the paper's training machines.

Preprocessing uses OpenCV for image operations, DuckDB and pyometiff for Xenium, and
SpatialData, spatialdata-io and pyvips for Visium HD. Downstream analysis uses rctd-py,
Tangram, Shapely, OpenSlide, PyArrow, natsort, NetworkX and Scanpy's Leiden dependencies.
Their version constraints are in the `preprocessing` and `downstream` extras in `setup.cfg`.
The optional `segmentation` extra names StarDist/TensorFlow dependencies for separately maintained
segmentation workflows; it does not implement mask generation in this package.
See [`data_preprocessing/README.md`](data_preprocessing/README.md#2-requirements).

### Tested system and hardware

The demo below was tested on **Ubuntu 20.04.6 LTS**, Python **3.11.15**,
PyTorch **2.12.0+cu126**, torchvision **0.27.0+cu126**, NVIDIA driver **570.133.20**, and one
**NVIDIA RTX 6000 Ada Generation (48 GiB)** on CB4. The supplied `FOCUS.yaml` describes a
different environment (Python 3.10.19 / PyTorch 2.9.0).

The other direct runtime dependencies tested were NumPy 2.4.4, SciPy 1.17.1, Scanpy 1.11.5,
AnnData 0.12.16, Transformers 5.9.0, POT 0.9.7.post1, blobfile 3.3.0,
mpi4py-mpich 3.1.5, scikit-image 0.26.0, PyYAML 6.0.3, pandas 2.3.3,
matplotlib 3.10.9, h5py 3.16.0 and Pillow 12.2.0. The full environment is recorded in
[`docs/environments/linux-cb4.txt`](docs/environments/linux-cb4.txt).

The Windows installation was tested in a new Python **3.12.14** environment on an
**ASUS TUF Gaming A15**, with an **AMD Ryzen 9 7940H**, **48 GB RAM** and
**Windows 10**. It used PyTorch
**2.9.0+cu126**, torchvision **0.24.0+cu126** and Intel MPI **2021.18.1**.
Core installation took **14 min 31 s**, including package downloads and one download retry;
Python, driver installation and model downloads were excluded. Core package versions are in
[`docs/environments/windows-core.txt`](docs/environments/windows-core.txt).

The preprocessing and downstream extras were installed in the same environment. All 39 tested
modules imported successfully; OpenCV affine transforms, libvips image creation and Parquet
read/write checks passed. Native library versions were libvips **8.18.7** and OpenSlide **4.0.1**.
The final package list is in [`docs/environments/windows-tools.txt`](docs/environments/windows-tools.txt).

Optional legacy paths use `accimage` for image decoding and TensorFlow 1.x for TensorBoard
logging. The tests above used Pillow decoding and the default stdout/log/CSV logger.

The model requires an NVIDIA CUDA GPU. An archived one-gene demo ran on this laptop's
**GeForce RTX 4060 Laptop GPU (8 GB VRAM)**, with NVIDIA driver **616.56** and the
PyTorch CUDA 12.6 build listed above.

### Demo

Run from the repository root after installing dependencies:

```bash
python demos/run.py
```

The bundled five-gene weight (`demos/focus-5gene.pt`) reads **one real-condition,
registered** H&E/ST tile from the public 10x Xenium mouse-pup sample. It jointly
generates **Ttn, Car3, Des, Cav1 and Myoz2** using 1,000 DDIM reverse steps by
default. `--steps N` is an explicitly shortened program check. The expected output
is `demos/output/3_10.npz` with one finite `(5,256,256)` float32 prediction and
`demos/output/3_10.png` with five grayscale panels. Measured transcript counts
are separately stored in `demos/input/3_10.npz`; predictions are normalized
expression values, not counts.

The validated 1,000-step run took **148.02 s** on one CB4 RTX 6000 Ada GPU,
including model loading and output writing. PyTorch reported **0.43 GiB peak allocated**
and **0.70 GiB peak reserved** CUDA memory; these are allocator figures on a 48 GiB card,
not a minimum VRAM requirement. Its preview is visibly speckled;
the 100-update checkpoint has no demonstrated biological reconstruction accuracy.

The real inputs contain official 10x cell and nuclear polygons, native H&E morphology
complexity, a P0 muscle scRNA reference from an independent study, and genuine
features extracted with official scGPT-spatial, Prov-GigaPath and frozen BioBERT
weights. The existing gene co-expression prior was derived from this training
tile. The checkpoint was trained from random initialization for **100 optimizer
updates with batch size 1 on this same tile**. This demonstrates a complete
five-gene path under real conditions; it is not an independent test or recovered
manuscript checkpoint. [Demo details](docs/usage.md) and [exact source and
preprocessing instructions](docs/real_data.md) distinguish the observed data,
model output, access requirements and known limits.

### Documentation and contribution

See [docs/](docs/index.md) for real data acquisition, input/output definitions, a source-aligned algorithm outline and
[model implementation and validation](docs/implementation.md). Contribution and release information
are in [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

If you are trying FOCUS for the first time, follow the installation and demo instructions above
and share your results through the [first-run feedback form](https://github.com/LHY1007/FOCUS/issues/new?template=first-run.yml).
The form asks about your computer, installation time, demo outputs and any unclear instructions.

GitHub Actions builds and checks the package on CPU. `.travis.yml`, `.pypirc` and `netlify.toml`
are optional configuration files for Travis CI, PyPI and Netlify; these services are not enabled.

The optional container recipe can be built with `docker build -t focus-st:0.3.2 .` and run with
`docker run --rm --gpus all -v "$PWD/demos/output:/opt/FOCUS/demos/output" focus-st:0.3.2`.
The container requires the NVIDIA Container Toolkit.

### Generated channels

Both the demo and the manuscript configurations generate **five genes** per group,
with output tensors `(batch, 5, 256, 256)`. H&E images retain their three RGB channels.
The manuscript configuration predicts five noise channels with a fixed diffusion variance,
as specified in Methods 1.1. Target-gene count is independent of H&E or encoder-feature dimensions.

## Data Preparation

### Directory Structure

FOCUS reads one directory tree per tissue type. Organize your data following this structure:

```
data_root/
├── {tissue_type}/
│   ├── {sample_id}/
│   │   ├── HR_ST/
│   │   │   └── extract/{layer}/{patch_id}/
│   │   │       └── HR_ST_256.npz
│   │   ├── spot_ST/
│   │   │   └── extract/{layer}/{patch_id}/
│   │   │       ├── spot_ST.npz
│   │   │       └── scgpt_data/
│   │   ├── WSI/
│   │   │   └── extract/{layer}/{patch_id}/
│   │   │       ├── 5120_to256.npy
│   │   │       ├── 320_to16.npy
│   │   │       ├── cell_mask.npy
│   │   │       ├── complexity_grid.npy
│   │   │       └── pre_he.npy
│   │   ├── gene_order.npy
│   │   └── gene_name.txt
│   ├── sc/
│   │   └── {scRNA_data}.h5ad
│   ├── coexpression_matrix.npy
│   └── coexpression_matrix.genes.txt
```

### Required Files

| File | Shape / format | Description |
| --- | --- | --- |
| `HR_ST_256.npz` | sparse `(65536, G)` | High-resolution ST on a 256×256 grid at 10 µm/bin |
| `spot_ST.npz` | sparse `(676, G)` | Low-resolution ST on a 26×26 grid at 100 µm/bin |
| `5120_to256.npy` | `(256, 256, 3)` uint8 | Registered H&E at 10 µm/px, downsampled from the native 0.5 µm/px tile |
| `320_to16.npy` | `(256, 16, 16, 3)` uint8 | Historical preprocessing output; the full loader rebuilds the cellular grid in row-major order from the registered H&E image |
| `cell_mask.npy` | `(256, 256, M)` | Cell segmentation masks |
| `complexity_grid.npy` | `(16, 16)` float32 | Density and native H&E nuclear shape, size and texture score from existing nuclear-instance labels |
| `pre_he.npy` | `(257, 1536)` | Authentic Prov-GigaPath vectors: tissue tile followed by 256 row-major cellular tiles |
| `scgpt_data/1to5.npy` | `(512, 26, 26)` | Authentic scGPT-spatial spot features; companion `1to5.genes.txt` identifies the five selected genes |
| `{scRNA_data}.h5ad` | AnnData | Single-cell reference, one `.h5ad` per tissue type |
| `coexpression_matrix.npy` | `(K, K)` | Spatial gene–gene co-expression prior for the tissue |
| `coexpression_matrix.genes.txt` | `K` unique names | Exact row and column gene order of that matrix; required to select current five-gene rows |
| `gene_order.npy` | `(G,)` int | Column indices into the platform's full gene panel |
| `gene_name.txt` | `(G,)` str | The matching gene names |
| `st_resolution.json` | JSON per sample | Measured pre-degradation ST bin size in µm and its physical source; required for the resolution gate |

`G` is the panel size of the sample, and it must match the gene dimension of `HR_ST_256.npz` and
`spot_ST.npz` (see [`data_preprocessing/README.md`](data_preprocessing/README.md#1-the-output-layout)).
The loader selects `gene_order[gene_start:gene_start+5]` and the corresponding names.
For each group it selects the scRNA reference **by those exact names and in that order**;
a missing or duplicate name raises an error. Groups use `1to5.npy`, `6to10.npy`, and so on.
`gene_name.txt[i]` must name the ST matrix column `gene_order.npy[i]`.
The loader also selects the current gene rows from the supplied co-expression matrix by name;
it never assumes that the first 50 matrix rows describe every gene group. Prepare the `.genes.txt`
sidecar from the exact gene order used to build that matrix; do not reorder matrix values.
Score `complexity_grid.npy` from the registered native H&E and existing nuclear labels using
[`data_preprocessing/score_complexity.py`](data_preprocessing/score_complexity.py). The downsampled
`cell_mask.npy` is insufficient for native subcellular shape and texture measurement.

The `sc/` directory must contain **exactly one** `.h5ad` — `load_dataset.py` raises if it finds zero or
more than one.

### Preprocessing Code

Platform-specific H&E/ST preprocessing lives in [`data_preprocessing/`](data_preprocessing/); existing masks, co-expression priors and pretrained features are supplied through separate steps.
It covers three of the ten evaluated platforms:

| Platform | Script | Produces |
| --- | --- | --- |
| 10x Visium | [`data_preprocessing/visium/visium.py`](data_preprocessing/visium/visium.py) | `5120_to256.npy`, `320_to16.npy`, `spot_ST.npz` |
| 10x Xenium | [`data_preprocessing/xenium/Xenium_pami.py`](data_preprocessing/xenium/Xenium_pami.py) | the above plus `5120.npy`, `HR_ST_256.npz`, `HR_ST_128.npz` |
| 10x Visium HD | [`data_preprocessing/visium_hd/`](data_preprocessing/visium_hd/) (8 steps) | the same as Xenium, plus `WSI_tile.png` and `gene_names.csv` |

All three pipelines share the same tile geometry — a 5120×5120 px tile at 0.5 µm/px (2560 µm of tissue) on a
50 % overlapping grid, with transcripts binned to 256×256 (10 µm) and 26×26 (100 µm) grids — which is what
makes the platforms interchangeable for the dataloader. `data_preprocessing/xenium/jsontocsv.py`
converts a Xenium `gene_panel.json` into the `gene_groups.csv` panel that defines the gene-column order.
Visium HD has no shared panel: it derives its gene order from the sample's own bin table and writes it to
`gene_names.csv`.

The per-platform inputs, configuration variables, run order and known caveats are documented in
[`data_preprocessing/README.md`](data_preprocessing/README.md).

Additional inputs have separate generators or export helpers:

| File | Produced by |
| --- | --- |
| `cell_mask.npy` | Existing cell instances registered to the patch; the public Xenium example uses official cell boundaries |
| `pre_he.npy` | Official [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) H&E features exported by `data_preprocessing/export_conditioning.py` after gated access |
| `scgpt_data/` | Official [scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial) embeddings exported by `data_preprocessing/export_conditioning.py` with verified checkpoint loading |
| `coexpression_matrix.npy` | spatial gene–gene co-expression computed across the cohort |
| `gene_order.npy`, `gene_name.txt` | gene indices/names derived from `gene_groups.csv` |

For the embedding and feature-extraction steps, refer to:

- [BioBERT](https://github.com/dmis-lab/biobert) — gene name embeddings
- [scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial) — gene expression embeddings
- [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath) — H&E feature extraction

Expression values are log-normalized with gene-wise scaling before training; `load_dataset.py` applies
min–max normalization to `spot_ST` and divides `HR_ST` by its per-patch maximum, keeping that maximum as
`patch_scale` so predictions can be mapped back to counts.

### Downstream Analysis

[`downstream/`](downstream/) holds the analyses that run on a spatial section once the patch layout
exists, and that produce the cell-type and trajectory results reported in the paper:

| Analysis | Path | What it does |
| --- | --- | --- |
| Annotation | [`downstream/annotation/scanpy_annotation.py`](downstream/annotation/scanpy_annotation.py) | **the annotation the results use**: highly variable genes, PCA, a kNN graph, UMAP and Leiden clustering, then Wilcoxon marker ranking for manual assignment |
| Deconvolution | [`downstream/deconvolution/`](downstream/deconvolution/) | **baseline replica**: builds a shared scRNA-seq reference and runs RCTD in doublet mode, then checks composition, markers and agreement against an independent run |
| Tangram baseline | [`downstream/annotation/tangram_reference.py`](downstream/annotation/tangram_reference.py) | **baseline replica**: maps a T2DM reference onto the section for the sensitivity comparison |
| Bin-to-cell matrix | [`downstream/matrix/bin2cell.py`](downstream/matrix/bin2cell.py) | aggregates Visium HD 2 µm bins into segmented nuclei with an STRtree point-in-polygon query |
| Trajectory inference | [`downstream/trajectory/`](downstream/trajectory/) | preprocesses and scores the section, then runs the stLearn PSTS trajectory (reimplemented) for the osteogenic, adipogenic and chondrogenic lineages |

The annotation route and the baseline route are separate: the RCTD and Tangram code answers the
benchmark comparisons, and nothing in `downstream/deconvolution/` feeds the annotation reported in
the paper. [`downstream/README.md`](downstream/README.md) spells out which is which.

These scripts are independent of training: they read the project's intermediate objects and write to
their own `output/` directories. Inputs, environment, run order and the known limits of each analysis
are documented in [`downstream/README.md`](downstream/README.md).

### Curated Cross-platform Multimodal Dataset

The manuscript reports **6,788 paired ST–H&E images** (over **1.7 million patches**) and over **5.8 million single-cell profiles** across ten ST platforms and 34 tissue contexts. Those data are not bundled in this repository. The link below points to the authors' external data location; check its current contents and permissions before planning a reproduction.

The fully specified, tested mouse-pup route and exact public entrances are in [real data and preparation](docs/real_data.md). For other cohorts, organize accessible samples into the directory structure above and provide:

- `HR_ST/` and `spot_ST/` — high- and low-resolution ST, with per-gene-group scGPT embeddings under `scgpt_data/`
- `WSI/` — paired H&E patches at 20× and at multi-scale, existing cell segmentation masks, native morphology complexity grids and pre-extracted H&E features
- `gene_order.npy` / `gene_name.txt` — original source channel indices and corresponding ordered gene names for each sample
- `sc/{scRNA_data}.h5ad` — an independent species, stage and tissue matched raw-count single-cell reference
- `coexpression_matrix.npy` and ordered `coexpression_matrix.genes.txt` — the spatial gene–gene co-expression prior and its row names

**External data location**: [curated cross-platform multimodal datasets](https://www.dropbox.com/scl/fo/8jipnyir34f1farrcgoxk/ADJbjaWY4V2Qv006H7j2j7w?rlkey=ca4buoejjn2d53rqjl5bii7yj&st=i68hdo96&dl=0)

Per-sample provenance (platform, tissue, accession) is listed in Supplementary Tables S1–S3 of the paper. The in-house papillary craniopharyngioma (PCP) datasets — Xenium spatial transcriptomics and CODEX spatial proteomics — are deposited in the National Genomics Data Center (NGDC) under accessions **OMIX021475** and **OMIX021470**, respectively.

### Registering a New Tissue Type

Use a new tissue directory with the same sample layout; `dataset_use: All` discovers it automatically.
Supply one tissue-matched raw-count scRNA reference in `sc/`, unique gene names, and real metadata
in `{tissue}/{sample}/metadata_prompt.txt`. This text identifies the actual platform, species and tissue;
the loader prefixes the current five gene names. Place the pinned official BioBERT v1.1 files
under `{pre_model_dir}/bert` as specified in [usage.md](docs/usage.md#prepare-a-new-experiment);
the loader rejects a generic BERT checkpoint. Reference cell sampling is deterministic (`reference_cells: 256`, `seed: 0`);
all reference genes are retained until name-based selection. For smaller references, set the same
positive `reference_cells` value no larger than the smallest available reference.

## Usage

Run the commands below from the repository root. By default, `train.py` and `sample.py` read
`config/config_train.yaml` and `config/config_test.yaml` beside the scripts. Use `--config`
to select another configuration. Data and output paths are relative to the working directory.

### Training

The default configuration follows Methods 1.1 and 3.1
and Supplementary Table S9:

```yaml
manuscript_mode: true
gene_num: 5
all_gene: 5
batch_size: 4
epoch: 2000
lr: 0.0001
diffusion_steps: 1000
timestep_respacing: ''
learn_sigma: false
predict_xstart: false
lambda_charb: 0.15
lambda_struct: 0.20
lambda_dis: 0.10
lambda_consis: 0.10
charbonnier_gamma: 0.0001
```

Create `data/samples.csv` with actual `tissue,sample_id,patient_id,split` values before extracting patches.
Use `train`, `validation`, `finetune` or `test` for the split. A patient may appear in only one split.
To create a new deterministic 80/10/10 assignment from a CSV with empty split cells:

```bash
python -m guided_diffusion.splits data/samples_unsplit.csv --assign --output data/samples.csv --seed 0
python -m guided_diffusion.splits data/samples.csv
python train.py --data_root data/prepared --split_manifest data/samples.csv --split train
```

The loader selects only the requested samples. A missing manifest entry or patient crossing splits is
an error. These commands create a new experiment's split; the original manuscript patient lists have
not been recovered. Use the original lists when repeating that experiment.

Each epoch visits the complete selected training set, including its final partial batch.
Each five-gene group receives the corresponding gene offset and writes its actual `config.yaml`
beside its checkpoints. `all_gene` must be a multiple of five; use a separate panel definition for a remainder.
To use multiple visible GPUs, launch with `torchrun --nproc_per_node=N train.py` and the same
configuration arguments. Ranks read patch shards and synchronize gradients;
the sampler repeats a few patches when their number is not divisible by the worker
count. Each rank writes a loss log, while rank zero writes model and optimizer
checkpoints.
`batch_size` is per rank, so the effective global batch is `N * batch_size`.
For one GPU, use `python train.py` as above. The reported seven-V100 manuscript run
and its global batch interpretation were not reconstructed.

### Fine-tuning

```bash
python train.py --config config/config_finetune.yaml --init_checkpoint models/foundation.pt --split finetune
```

The U-Net encoders and time embedding are frozen. Decoder and conditional matching/fusion layers
remain trainable. Task-module weight matrices receive low-rank updates while their base weights stay
frozen. Rank 8 and zero-initialized updates are implementation defaults; the manuscript does not specify
the rank. Adapted checkpoints can resume with `--resume_checkpoint` pointing to
`modelNNNNNN.pt` in its `Gstart-end_*` run directory, together with `optNNNNNN.pt`;
the loop loads both states and continues up to the configured total epoch count.
`merged.pt` exports ordinary
weights for inference. `--tasks sensitivity,alignment,cell_fidelity` controls available tasks;
absent tasks are disabled before expert execution. Per-sample activation uses the paper's `score > 0.5` rule.

### Inference

```bash
python sample.py --config config/config_test.yaml --data_root data/prepared --split_manifest data/samples.csv --split test
```

The default formal reverse process uses 1,000 DDPM steps; set `sampling_method: ddim` to use DDIM.
`sample.py` reads five-gene checkpoints under
`{checkpoint_root}/{dataset_use}/{SR_times}X/G*/`, preferring `merged.pt` for adapted models.
It writes `(256,256,5)` expression arrays and computes RMSE, mean per-gene SSIM and PCC directly
on normalized predictions without applying a separate per-image min/max transformation.

### Pre-trained Models

The five-gene software-demo checkpoint is bundled in `demos/` and uses the matching input conditions
described in [usage.md](docs/usage.md). Historical checkpoints are available from
[Dropbox](https://www.dropbox.com/scl/fo/gte7fbz2y14syitka3mb0/AOhsVHx4Rdlk9BU2oJRySv4?rlkey=aytpdtg1ae05jf8i139a2e15c&st=5qi9943t&dl=0).
Check their architecture and gene order before use; they are not relabeled as weights for the new implementation.

## Evaluation Metrics

The equation mapping, implementation choices and verification scope are listed
in [reproduction notes](docs/reproduction.md). The demo uses its own checkpoint and sampling settings.

The built-in `sample.py` reports only the following patch-level metrics:

- **RMSE**: root mean squared error for intensity accuracy
- **SSIM**: structural similarity index for spatial patterns
- **CC / PCC**: Pearson correlation coefficient for gene-wise concordance

`sample.py` writes these three per patch in `metrics.csv` and a JSON record linking each
prediction to its sample, gene order and physical ST resolution. For a paired five-gene NPZ:

```bash
python -m evaluation.evaluate_pair --observed demos/input/3_10.npz \
  --prediction demos/output/3_10.npz --input-baseline \
  --output demos/output/metrics.json
```

This exports normalized RMSE, SSIM and PCC, observed UMI totals, gradient LNCC and
gradient SSIM against the registered H&E, RMSE by native-complexity tercile,
and a count-conserving interpolation of
the observed low-resolution input for comparison. It does not convert normalized
predictions into UMI counts. The bundled pair is a training-tile reconstruction;
Add `--compare label=/path/to/prediction.npz` for another run on the same tile and
gene order; each named run is scored independently. External baselines and
manuscript figure estimates require their own matched data.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use gradient checkpointing: `use_checkpoint: true`
   - Enable mixed precision: `use_fp16: true`

2. **Missing Gene Files**
   - Check that `gene_order.npy` and `gene_name.txt` exist for each sample
   - Verify gene indices match across ST and scRNA-seq data
   - Confirm the tissue directory and its sample ID match `samples.csv`; the loader discovers tissue names from the selected directories
   - `HR_ST` and `spot_ST` must have the same gene dimension, since both are indexed with `gene_order`

3. **Configuration file not found**
   - The default config is resolved relative to the entry-point file. Use `--config /path/to/config.yaml`
     for another configuration; data and output paths are relative to the working directory.

4. **No patches loaded**
   - The dataset requires **all** of `HR_ST_256.npz`, `spot_ST.npz`, `5120_to256.npy`, `320_to16.npy`
     and `cell_mask.npy` for a patch; a patch missing any of them is skipped with a warning
   - Supply `pre_he.npy` and `scgpt_data/` as well: the current model requires numeric tensors for both.
     The bundled demo supplies its precomputed conditions; see [input definitions](docs/usage.md).
   - Tiles rejected by the H&E pass can still have ST grids written for them (see
     [`data_preprocessing/README.md`](data_preprocessing/README.md#55-run-order)); those tiles have no
     `WSI/` patch and are skipped here

5. **Slow Training**
   - Adjust `num_workers` in the `DataLoader`
   - Use SSD storage for data
   - Enable `persistent_workers: true`

6. **NaN Losses**
   - Check input data normalization
   - Verify no inf/nan values in co-expression matrices
   - Reduce learning rate

## Citation

If you use FOCUS in your research, please cite:

```bibtex
@article {Wang2026Focus,
	author = {Wang, Xiaofei and Que, Ningfeng and Liu, Hanyu and Tao, Chenyang and Jiang, Yu and Jiang, Yixuan and Zhu, Pinan and Xu, Jianguo and Yan, Huachao and Ye, Buqing and Price, Stephen and Xi, Jianzhong Jeff and Wang, Xinjie and Li, Chao},
	title = {Coordinated Enhancement of Molecular and Spatial Fidelity for Biological Inference in Spatial Transcriptomics},
	elocation-id = {2025.12.23.696267},
	journal = {bioRxiv},
	year = {2026},
	doi = {10.64898/2025.12.23.696267},
}
```

## Data and Code Availability

- **Code**: https://github.com/LHY1007/FOCUS
- **Curated cross-platform multimodal datasets** (paired ST–H&E images, scRNA-seq references, segmented cell masks and pre-computed spatial gene co-expression matrices): [Dropbox](https://www.dropbox.com/scl/fo/8jipnyir34f1farrcgoxk/ADJbjaWY4V2Qv006H7j2j7w?rlkey=ca4buoejjn2d53rqjl5bii7yj&st=i68hdo96&dl=0), with per-sample provenance summarized in Supplementary Tables S1–S3
- **In-house PCP datasets** (Xenium spatial transcriptomics and CODEX spatial proteomics): National Genomics Data Center, accessions OMIX021475 and OMIX021470

## Acknowledgments

FOCUS uses code and models from:

- **Diffusion Models**: [Improved DDPM](https://github.com/openai/improved-diffusion)
- **BioBERT**: [dmis-lab/biobert](https://github.com/dmis-lab/biobert)
- **scGPT**: [bowang-lab/scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial)
- **GigaPath**: [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath)

## Contact

For questions and feedback:
- Open an issue on GitHub
- Email: xw405@cam.ac.uk
