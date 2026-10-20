# Inputs and outputs

## Five-gene real-condition software demo

```bash
python demos/run.py
```

The runner strictly loads the bundled `demos/focus-5gene.pt` and one registered
`demos/input/3_10.png` / `3_10.npz` pair. It generates **Ttn, Car3, Des, Cav1,
Myoz2** jointly using 1,000 DDIM reverse steps by default; `--steps N` explicitly
shortens a program check. The H&E and observed ST are from the **same** field in
10x's public P1 Xenium mouse-pup sample. The checkpoint was fitted on this very
tile for 100 optimizer updates from random initialization, batch 1.

The genuine conditions are an independent P0 mouse muscle scRNA reference (256
cells from two donors), official 10x cell polygons (68 selected occupancy planes),
native H&E plus official nuclear-boundary complexity, official scGPT-spatial and Prov-GigaPath
pretrained embeddings, frozen official BioBERT v1.1 metadata, and a read-only
50-gene co-expression prior derived from the **training tile**. The selected
masks omit most cells; the scGPT human vocabulary needs explicit mouse-to-human
ortholog mapping. The prior is in-sample. Exact access, hashes, selection and
limits are in [real_data.md](real_data.md) and `demos/input/provenance.json`.

| Input NPZ key | Shape before batching | Meaning |
| --- | --- | --- |
| `low_res` | (5,26,26) | Normalized five-gene input from observed low-resolution counts |
| `observed_hr_counts`, `observed_low_counts` | (5,256,256), (5,26,26) | Actual measured counts; **not** model predictions |
| `WSI_320` | (256,3,16,16) | Registered row-major H&E cellular tile grid |
| `Gene_index_map`, `gene_class` | (5,256,256), (5,) | Five selected source gene identities |
| `metadata_feature` | (768,) | Frozen BioBERT encoding of true sample metadata and current gene names |
| `co_expression` | (5,64) | Current five named rows pooled from the existing same-tile ST prior |
| `WSI_mask` | (68,256,256) | Selected official 10x cell polygon occupancy at 10 µm/pixel |
| `complexity` | (16,16) | Native H&E + official nuclear-label density/size/shape/texture complexity |
| `sc` | (256,5) | Independent P0 muscle reference, aligned by exact names |
| `scgpt` | (512,26,26) | Official scGPT-spatial v1 encoding of the measured ST with verified checkpoint loading |
| `pre_he` | (257,1536) | Official Prov-GigaPath tile embeddings from native H&E |
| `scale_gt` | scalar | Legacy compatibility input, not used as count ground truth |
| `genes` | (5,) string | Exact target order |
| `original_st_bin_um` | scalar | Measured pre-degradation ST bin size, in µm, from sample provenance |

`demos/output/3_10.npz` stores finite float32 `prediction` with shape
(5,256,256), the five names, 1,000-step setting and “training-tile
reconstruction” label. Its PNG is a 256×1280 five-panel grayscale strip:
`round(255*clip(prediction,0,1))` joined in gene order. These normalized model
values are not transcript counts. Check `demos/input/3_10.npz` for the separately
stored observed count arrays. Only one pair is bundled to keep the example small.
The validated 1,000-step run took 148.02 s on one CB4 RTX 6000 Ada 48 GiB GPU,
including loading and writing. Peak CUDA memory was 0.43 GiB allocated and 0.70 GiB
reserved by PyTorch; this is not a measured minimum VRAM requirement. Its
preview is visibly speckled; do not use it as evidence of expression fidelity.

To compare this paired output with the observed five-gene counts, run:

```bash
python -m evaluation.evaluate_pair --observed demos/input/3_10.npz --prediction demos/output/3_10.npz --input-baseline --output demos/output/metrics.json
```

The JSON reports each gene separately, the observed UMI total, and normalized RMSE,
SSIM and PCC, plus registered H&E gradient LNCC, gradient SSIM and RMSE
by native-complexity tercile. The optional
input baseline spreads each observed 100 um count across its contributing 10 um
bins and preserves per-gene totals. Normalized model values are not UMI counts.
The demo reconstructs its training tile, so these metrics are not independent
accuracy estimates. Constant channels have undefined PCC (`null`).

## Prepare a new experiment

1. Record real tissue, sample_id, patient_id and split assignments before patch extraction. Validate
   with `python -m guided_diffusion.splits data/samples.csv`. All sections from one patient retain
   the same ID and split. Keep external testing and fine-tuning specimens separate.
2. Register H&E and ST in physical coordinates. Supplementary Section 6 defines 2,560x2,560 um fields,
   H&E stored at 0.5 um/pixel and downsampled to 10 um/pixel, HR ST at 10 um/pixel and LQ ST at
   100 um/pixel. For alignment evaluation, perturb only after reference registration and exclude
   reference landmarks/transforms from model inputs. Platform raw inputs are described in
   `data_preprocessing/README.md`.
   Record the measured **pre-degradation** ST bin width and its derivation in
   `{tissue}/{sample}/st_resolution.json`, for example
   `{"original_st_bin_um": 10.0, "source": "2560 um field / 256 HR bins"}`.
   The source bin width controls the resolution gate; the 26×26 model input
   dimensions do not substitute for physical measurement.
3. Apply the supplement's ST filtering: remove genes expressed in fewer than 2% of pixels; discard
   pixels expressing fewer than 2% of genes. Keep grid positions as zero rows and retain the same
   gene columns in HR and LQ maps. Record the retained names and column indices.
4. Supply existing cell masks and spatial co-expression input through the documented interface.
   Their generation is maintained separately. Supply a species/tissue-matched raw-count reference,
   perform QC, preserve unique gene names and store one h5ad per tissue in `sc/`. The loader
   normalizes, logs, samples cells deterministically and selects reference genes by exact name.
   Put one gene name per line in `coexpression_matrix.genes.txt`, in the matrix's exact row and
   column order. The loader extracts the current five rows by name and rejects missing names.
   Also provide registered native-resolution H&E and existing integer nuclear-instance labels for
   each patch. Score them into `WSI/extract/{layer}/{patch_id}/complexity_grid.npy` with:

   ```bash
   python data_preprocessing/score_complexity.py --he native_he.npy --labels native_nuclear_labels.npy --label-kind nucleus --pixel-size-um 0.5 --output /prepared/tissue/sample/WSI/extract/layer/patch/complexity_grid.npy
   ```

   The command measures existing labels; it does not segment cells. Each 2,560 µm field produces
   16×16 density plus shape, size and hematoxylin texture scores. The 256×256 mask alone is too
   coarse to preserve subcellular morphology at 10 µm/pixel, so full loading requires this grid.
5. Export authentic encoder features below. Full-dataset loading rejects missing and entirely zero
   features. The bundled demo uses nonzero official encoder features and native morphology from real inputs.
6. Save actual platform, species and tissue in each sample's `metadata_prompt.txt`. Place the pinned
   [dmis-lab/biobert-v1.1](https://huggingface.co/dmis-lab/biobert-v1.1) revision
   `551ca18efd7f052c8dfa0b01c94c2a8e68bc5488` in `models/bert` (config, vocabulary,
   tokenizer files and `pytorch_model.bin`). The loader verifies the model file SHA256
   `8e87cb27d889186c442feff2bea53f8cce2c5fc24d4bc88bf08ccd3821b138dc`, freezes
   its parameters and prefixes the current five gene names before tokenization. A generic
   768-dimensional BERT is rejected. Set root, manifest, split and checkpoint paths.

`gene_name.txt[i]` must name column `gene_order.npy[i]`. For offsets 0,5,10,... the loader
selects five names and reorders the reference accordingly. It does not match numeric positions or
reuse the first reference for other samples in a batch.

## Prov-GigaPath features

Use the original [tile encoder and its upstream environment](https://github.com/prov-gigapath/prov-gigapath#inference-with-the-tile-encoder).
The helper follows published RGB resize, crop and ImageNet normalization on the registered native
5120×5120 H&E field. It saves the 1536-dimensional tissue vector first, then 256 vectors from
native 320×320-pixel cellular tiles in row-major order. The downsampled 256×256 image would erase
subcellular texture before encoding and is rejected here.

```bash
python data_preprocessing/export_conditioning.py gigapath --he /prepared/tissue/sample/WSI/extract/layer/patch/5120.npy --output /prepared/tissue/sample/WSI/extract/layer/patch/pre_he.npy
```

## scGPT-spatial features

Use the [upstream environment](https://github.com/bowang-lab/scGPT-spatial) and its
[embed_data interface](https://github.com/bowang-lab/scGPT-spatial/blob/main/scgpt_spatial/tasks/cell_emb.py).
The checkpoint directory contains best_model.pt, args.json, vocab.json and all_dict_mean_std.csv.
The helper encodes selected five-gene nonempty rows and returns vectors to their actual 26x26
positions. Empty rows keep zero vectors. Set st_feature_dim to the checkpoint's actual embsize.

```bash
python data_preprocessing/export_conditioning.py scgpt --spot /prepared/tissue/sample/spot_ST/extract/layer/patch/spot_ST.npz --gene-order /prepared/tissue/sample/gene_order.npy --gene-names /prepared/tissue/sample/gene_name.txt --model-dir /models/scgpt-spatial --gene-start 0 --gene-num 5 --output /prepared/tissue/sample/spot_ST/extract/layer/patch/scgpt_data/1to5.npy
```

For the bundled mouse five genes, also pass `--ortholog-map data_preprocessing/examples/mouse_pup_scgpt_orthologs.csv` and apply the pinned upstream non-Flash patch described in [real_data.md](real_data.md). Use offset 5 and 6to10.npy for the next group. A companion .genes.txt records selected names;
the loader verifies their order. Output JSON records extraction arguments and the feature hash.
Existing features are not overwritten. Both official pretrained weight sets were actually loaded and used for the bundled tile; see [real_data.md](real_data.md) for the pinned versions, scGPT compatibility patch, matched-tensor checks and limits.
