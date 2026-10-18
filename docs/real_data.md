# Real data and reproducible input preparation

The bundled five-gene example is one **training tile** from one 10x mouse-pup donor. Its
H&E and observed Xenium counts cover the same registered 2,560 µm field. The independent
single-cell reference is age and tissue matched, but comes from another study and two
donors. No held-out accuracy can be estimated from this example. The demo's observed
count arrays are ground truth; `demos/output/` contains model predictions.

## Sources and access

| Input | Exact public entrance and version | Access and local handling |
| --- | --- | --- |
| H&E, transcripts, gene panel, alignment | [10x whole mouse pup, Xenium Mouse Tissue Atlassing Panel](https://www.10xgenomics.com/datasets/mouse-pup-preview-data-xenium-mouse-tissue-atlassing-panel-1-standard), Xenium Onboard Analysis 1.6.0, one P1 sagittal section | CC BY 4.0. Download the output and supplemental aligned post-Xenium H&E. The direct `Xenium_V1_mouse_pup_he_image.ome.tif` and `Xenium_V1_mouse_pup_transcripts.parquet` files are under `https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_mouse_pup/`. Keep these multi-GB files outside Git. Use the matching `he_imagealignment.csv`, `experiment.xenium`, and `gene_panel.json` from the same release. |
| Existing cell and nucleus segmentations | `Xenium_V1_mouse_pup_cell_boundaries.parquet` and `Xenium_V1_mouse_pup_nucleus_boundaries.parquet` at the same 10x S3 prefix | CC BY 4.0. These are the vendor's cell and nucleus polygons, not H&E segmentations produced by FOCUS. The 68 selected cell occupancy planes preserve official cell IDs; the separate nuclear labels measure the paper's morphology descriptors. |
| Independent scRNA | [Cao et al., mouse developmental atlas](https://cellxgene.cziscience.com/collections/45d5d2c3-bc28-4814-aed6-0bb6f0e11c82), “Major cell cluster: Muscle cells”, dataset `eaa7a0b0-36c9-4ec3-8023-d999eeeed441`, [H5AD asset](https://datasets.cellxgene.cziscience.com/d33cc838-a0e3-40f7-ae99-0384b13db313.h5ad) | Download under CELLxGENE's displayed data terms; cite Cao et al. The example selects 256 P0/Theiler 27 cells, seed 0, from the **integer `raw.X` layer**, retaining all 45,525 genes. No Xenium row is relabeled as an scRNA cell. |
| scGPT-spatial | [Official v1 Figshare weights](https://figshare.com/articles/code/scGPT-spatial_V1_Model_Weights/28356068), [source](https://github.com/bowang-lab/scGPT-spatial) commit `f9442d777ba47cd3dcac8253833a233ff5bf7262` | Weights CC BY 4.0; source MIT. The five mouse symbols require the explicit ortholog CSV in `data_preprocessing/examples/`. The official human vocabulary cannot accept their mouse names directly. |
| Prov-GigaPath | [Official tile encoder](https://huggingface.co/prov-gigapath/prov-gigapath), revision `64f9e26c15019f2d4f6d9113c6822f88bb16b01b` | Apache 2.0; Hugging Face login and acceptance of the model's access terms are required. Store its ~4.5 GB weights in the Hugging Face cache outside Git. |
| BioBERT | [Official `dmis-lab/biobert-v1.1`](https://huggingface.co/dmis-lab/biobert-v1.1), revision `551ca18efd7f052c8dfa0b01c94c2a8e68bc5488` | Download model, config, vocabulary and tokenizer files to `{pre_model_dir}/bert`; the loader verifies the PyTorch weight SHA256 `8e87cb27d889186c442feff2bea53f8cce2c5fc24d4bc88bf08ccd3821b138dc`. Check the model page for its current usage terms. |
| Spatial co-expression prior | For the bundled example, an **existing** 50-gene matrix from this same mouse-pup tile, with `coexpression_matrix.genes.txt` naming rows and columns | The matrix was copied read-only; its generation is maintained separately. It is an in-sample prior and unsuitable as evidence of independent generalization. New cohorts need an appropriately derived prior supplied by its owner. |

Downloaded source SHA256 values for this run were `476b6d94c56bdabd5db010d5f54b7bf46e4b6780ac1dfc98812d84bc5b0e8f6a`
(10x H&E OME-TIFF), `c5e56694cea0cdbe2d3c191ce837fe07b0906de2d90bb1b23dc0999752d490d4`
(10x transcript Parquet), `524c8b1da49de09a70a0b8d3283157d0a4c91f599da16996bf23cd452605d5f8`
(cell polygons) and `9fa5691709f86c3edc746d03d5d81b9313c1d3914caa173d6e879d7783c95762`
(nuclear polygons). Official scGPT-spatial `best_model.pt` SHA256 was
`1ce6a9eaa7cc199b23d404ea38203956d04699aaede42b6ea67aa0bf3a88c8be`;
the Prov-GigaPath PyTorch weight SHA256 was
`877947214318afa9e011754b74bbc3894a1f480a253afc7bc8045b8321dedd63`.

## Processed example and validation

The pre-existing registered 10x tree was read at `Preview_379/processed`. Its tile `3_10`
has top-left morphology coordinates `(3840, 12800)` µm. Native H&E is 5,120×5,120 RGB
at 0.5 µm/pixel; 256×256 H&E and HR ST are at 10 µm/bin, and 26×26 low-resolution ST
is at 100 µm/bin. Original 399-column matrices use columns **365, 52, 116, 53, 234**
for **Ttn, Car3, Des, Cav1, Myoz2**. HR and low-resolution observed count totals agree
gene by gene: **755420, 546715, 536507, 350435, 332624**. Neither the five channels
nor the reference genes are selected by taking the first five columns.

`Preview_379/st_resolution.json` records `original_st_bin_um: 10.0`, calculated
from the registered **pre-degradation** `HR_ST_256.npz` grid (2,560 µm / 256
bins), with that source stated in the file. Raw Xenium transcript coordinates
are finer than this prepared grid. The task-resolution gate reads this numeric
sample metadata and activates only for values **greater than** 10 µm; the
26×26 conditioning tensor size does not determine it. New samples must carry
their own measured physical bin size and source; the loader provides no
platform-name fallback. The saved training config records the values for
checkpoint resume checks.

On this tile, `read_xenium_boundaries.py` projected official **nuclear** polygons
into a native label raster (46,036 intersect the field, 46,035 remain visible).
`score_complexity.py` used these nuclear labels and H&E to score density,
nuclear size, shape and hematoxylin texture in a 16×16 grid. A separate
projection of official **cell** polygons had 46,348 intersecting and 46,336
visible instances. `select_official_cells.py` packed 68 documented vendor cell
IDs into the 10 µm occupancy input. This selection omits most visible cells and the
co-expression prior uses the training tile; both limit biological conclusions. It
does not replace the separately maintained mask generator.

The scGPT-spatial export produced 512×26×26 features after matching **153/153**
required pretrained encoder tensors. Its non-Flash source needs the small, recorded
compatibility patch `data_preprocessing/scgpt_compat/official_v1_nonflash.patch` and
the local `torchtext` compatibility shim in the same folder with our PyTorch build.
Without that patch, 24 attention projection tensors silently fail to load; a feature
file from that state must not be used. The official Prov-GigaPath tile encoder made
257×1536 nonzero vectors from native H&E, including 256 row-major 320-pixel tiles.
BioBERT was frozen and encoded actual sample metadata plus the current five names.

The resulting nuclear-label formal tree was kept outside Git;
it passed `Data_all` strict loading and manifest validation with
`guided_diffusion.splits`. Run that validator on your own prepared `samples.csv`.
The tested manifest
has one `train` donor and no validation/test donor. Training used random initialization,
one tile, batch 1, learning rate 1e-4 and 100 optimizer updates, with the paper's four
loss weights. The run is a real-condition execution check, not the manuscript's
2,000-epoch or seven-GPU experiment. Its model, optimizer and configuration
were retained outside Git; the compatible inference checkpoint is bundled in `demos/`.
The saved configuration records the measured bin size and its physical derivation;
resume checks both. The physical source record was copied from the already present
`st_resolution.json` into that configuration after training.
`sample.py` also loaded the saved config/checkpoint and
ran on this formal tree with `split=train` and an explicitly shortened two-step
DDPM execution check. The bundled demo ran the full 1,000-step DDIM chain on the
same real conditions in 148.02 s on one RTX 6000 Ada GPU (0.43 GiB peak CUDA
allocated; 0.70 GiB peak reserved). Its prediction is visibly speckled, so neither
run establishes biological accuracy.
The paired evaluation command's count-conserving low-input interpolation has
normalized RMSE 0.0827 on this training tile, versus 0.5129 for the bundled
checkpoint. The demo checkpoint therefore does not outperform even this simple
input baseline on its fitted tile.

## Preparing your own formal tree

1. Download the **matching** `experiment.xenium`, `gene_panel.json` and
   `he_imagealignment.csv` into one sample directory outside the repository. Check
   their sample identity and 10x coordinates. For a single tile, the large original
   transcript and H&E objects can be read by HTTP range without downloading their
   entire 13 GB combined size:

   ```bash
   RAW=/path/to/Xenium_V1_mouse_pup/outs
   BASE=https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_mouse_pup
   python data_preprocessing/xenium/replay_registered_tile.py \
     --raw-dir "$RAW" \
     --he "$BASE/Xenium_V1_mouse_pup_he_image.ome.tif" \
     --transcripts "$BASE/Xenium_V1_mouse_pup_transcripts.parquet" \
     --tile 3_10 --output /path/to/replayed/3_10
   ```

   The replay writes all 399 panel channels, the source gene order, registered
   native H&E and both ST grid scales. It was compared to the separately prepared
   `Preview_379/processed` tile: all three sparse ST matrices agreed exactly,
   including 10,590,521 panel transcripts. OpenCV and libvips bilinear H&E
   resampling differed by at most three 8-bit levels in the native field and one
   level at 256×256. See the
   [preprocessing layout](../data_preprocessing/README.md) for bin conventions.
2. **Before patching**, assign all sections from each patient/donor to one split.
   Create `samples.csv` with `tissue,sample_id,patient_id,split`, then run
   `python -m guided_diffusion.splits /path/to/samples.csv`. Overlapping 50% tiles
   from one donor must never cross splits. This single-donor example is `train` only.
3. Place each observed HR/low ST and its registered H&E in the documented formal
   layout. Record the original gene panel and channel indices; `gene_name.txt[i]`
   must identify source column `gene_order.npy[i]`. Low- and high-resolution counts
   must use the same panel and physical field. In each sample directory create
   `st_resolution.json` with numeric `original_st_bin_um` and a nonempty
   `source` describing how the **pre-degradation** ST bin size was measured.
   For the bundled tile the value is 10.0 from 2,560 µm / 256 HR bins.
   `demos/prepare_observed.py` and
   `demos/assemble_real_data.py` show validated extraction and assembly for this tile.
4. Supply **existing** native cell and nuclear instance labels, plus a compatible
   prior from their maintainers. To turn existing **nuclear** labels into complexity, run
   `python data_preprocessing/score_complexity.py --he /path/to/native_he.npy
   --labels /path/to/native_nucleus_labels.npy --label-kind nucleus --pixel-size-um 0.5
   --output /path/to/complexity_grid.npy`. For official Xenium polygons, first use
   `data_preprocessing/read_xenium_boundaries.py` with the field's `--x-um`,
   `--y-um` and `--boundary-type nucleus`. Project the cell-boundary file
   separately with `--boundary-type cell`; `demos/select_official_cells.py`
   only selects its existing vendor IDs. Do not treat cell contours as nuclear
   shape measurements. The scorer writes source hashes in a `.json` sidecar.
   The model also requires `coexpression_matrix.genes.txt` in exact matrix order.
5. Download an independent tissue and stage matched raw-count scRNA reference.
   For this exact public example, run `data_preprocessing/extract_mouse_pup_reference.py
   --source /path/to/muscle_cluster.h5ad --output /path/to/p0_muscle_256.h5ad`.
   Keep source cell IDs, donors, selection seed, gene symbols and original count layer.
6. Cache the three official pretrained models at the pinned versions above. In the
   scGPT-spatial source checkout, apply the recorded patch with `patch -p1 <
   /path/to/FOCUS/data_preprocessing/scgpt_compat/official_v1_nonflash.patch`.
   Set `PYTHONPATH=/path/to/FOCUS/data_preprocessing/scgpt_compat:/path/to/scGPT-spatial`
   when running `data_preprocessing/export_conditioning.py scgpt`; pass `--spot`,
   `--gene-order`, `--gene-names`, `--model-dir`, `--gene-num 5`, `--gene-start 0`,
   `--ortholog-map data_preprocessing/examples/mouse_pup_scgpt_orthologs.csv`,
   `--output /path/to/scgpt_data/1to5.npy` and `--device cuda`. Inspect its JSON:
   all required checkpoint tensors must match. Run the same helper's `gigapath`
   mode with `--he /path/to/native_he.npy --output /path/to/pre_he.npy`.
   `CUDA_VISIBLE_DEVICES` should expose only a GPU allocated to this job.
7. Put the real platform, species, age and tissue in each sample's
   `metadata_prompt.txt`. Put BioBERT in `{pre_model_dir}/bert`. Validate the tree
   with `python -m guided_diffusion.splits /path/to/samples.csv` and a strict
   `Data_all` load; `demos/export_real_demo.py --data-root /path/to/prepared
   --pre-model-dir /path/to/pretrained --output /path/to/empty_demo_input`
   validates and exports this exact single-tile example. It refuses other samples.
   For your cohort, use `train.py --data_root /path/to/prepared
   --split_manifest /path/to/samples.csv --split train` and `sample.py` with the
   same root/manifest and an independent `--split test`. Configure five-gene groups
   and use the true target-gene order. A checkpoint trained on unrelated proxy
   conditions is not a substitute for fitting the genuine inputs.

The bounded 10x raw-to-tile replay was completed for tile `3_10`; the large raw
files were read directly from the official HTTP range source. The existing
registered tree supplied official feature extraction, strict formal loading,
100 real-condition updates and the bundled inference. A formal tree with the
raw-replayed H&E and both ST grids also passed strict loading and a two-step
DDPM run with the same checkpoint; it produced a finite 256×256×5 prediction.
The previously exported H&E features and nuclear complexity were retained from
the same registered sample; their source H&E crop differs from the new OpenCV
crop by the small resampling amounts stated above. No full 1,000-step run was
repeated on this replayed crop.
Original manuscript
patient lists, historical checkpoint and figure-specific data remain unavailable.
