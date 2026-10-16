# Inputs and outputs

## Five-gene software demo

```bash
python demos/run.py
```

The runner reads the bundled `demos/focus-5gene.pt` and ten H&E PNG/NPZ pairs.
It generates `Ttn, Car3, Des, Cav1, Myoz2` together with 1,000 DDIM reverse steps by default.
`--steps N` selects an explicitly shortened run. No checkpoint download is performed.

Each input PNG is registered RGB H&E, shape (256,256,3). Its NPZ contains:

| Key | Shape before batching | Meaning |
| --- | --- | --- |
| low_res | (5,26,26) | Five gene expression planes |
| WSI_320 | (256,3,16,16) | Row-major cellular tile grid |
| Gene_index_map | (5,256,256) | Gene identity planes |
| gene_class | (5,) | Gene indices |
| metadata_feature | (768,) | Frozen official BioBERT v1.1 encoding of the Xenium mouse-pup prompt and current five genes |
| co_expression | (K,K) | Existing gene-context input |
| WSI_mask | (M,256,256) | Existing proxy components |
| sc | (100,5) | Adult WMB reference in target-gene order |
| scgpt | (512,26,26) | Zero pretrained-feature placeholder in the demo |
| pre_he | (257,1536) | Zero pretrained-feature placeholder in the demo |
| scale_gt | scalar | Compatibility field; not used to reconstruct counts |
| genes | (5,) string | Target-gene order |

The H&E/ST measurements come from the public Xenium mouse-pup tiles used by the earlier demo.
The adult WMB reference differs from that tissue. Proxy masks and zero scGPT/GigaPath arrays are
retained as disclosed demo conditions. Each input identifies its source tile. The new weight was
trained from random initialization for 100 updates with batch size 4 and the manuscript's four-term
objective. It is not a recovered checkpoint from the paper experiments.

Ten pairs produce ten PNGs and ten NPZs in `demos/output/`. Each NPZ stores finite float32
`prediction`, shape (5,256,256), and `genes`. Each PNG is a 256x1280 grayscale strip: five panels
in that order, all using [0,1]. It is exactly `round(255*prediction)` joined horizontally.
Values are normalized expression, not transcript counts. Existing output stems are overwritten.

## Prepare a new experiment

1. Record real tissue, sample_id, patient_id and split assignments before patch extraction. Validate
   with `python -m guided_diffusion.splits data/samples.csv`. All sections from one patient retain
   the same ID and split. Keep external testing and fine-tuning specimens separate.
2. Register H&E and ST in physical coordinates. Supplementary Section 6 defines 2,560x2,560 um fields,
   H&E stored at 0.5 um/pixel and downsampled to 10 um/pixel, HR ST at 10 um/pixel and LQ ST at
   100 um/pixel. For alignment evaluation, perturb only after reference registration and exclude
   reference landmarks/transforms from model inputs. Platform raw inputs are described in
   `data_preprocessing/README.md`.
3. Apply the supplement's ST filtering: remove genes expressed in fewer than 2% of pixels; discard
   pixels expressing fewer than 2% of genes. Keep grid positions as zero rows and retain the same
   gene columns in HR and LQ maps. Record the retained names and column indices.
4. Supply existing cell masks and spatial co-expression input through the documented interface.
   Their generation is maintained separately. Supply a species/tissue-matched raw-count reference,
   perform QC, preserve unique gene names and store one h5ad per tissue in `sc/`. The loader
   normalizes, logs, samples cells deterministically and selects reference genes by exact name.
   Put one gene name per line in `coexpression_matrix.genes.txt`, in the matrix's exact row and
   column order. The loader extracts the current five rows by name and rejects missing names.
   Also provide registered native-resolution H&E and existing integer cell-instance labels for
   each patch. Score them into `WSI/extract/{layer}/{patch_id}/complexity_grid.npy` with:

   ```bash
   python data_preprocessing/score_complexity.py --he native_he.tif --labels native_cell_labels.npy --pixel-size-um 0.5 --output /prepared/tissue/sample/WSI/extract/layer/patch/complexity_grid.npy
   ```

   The command measures existing labels; it does not segment cells. Each 2,560 µm field produces
   16×16 density plus shape, size and hematoxylin texture scores. The 256×256 mask alone is too
   coarse to preserve subcellular morphology at 10 µm/pixel, so full loading requires this grid.
5. Export authentic encoder features below. Full-dataset loading rejects missing and entirely zero
   features. The software demo's placeholders are not used for full-dataset loading.
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

Use offset 5 and 6to10.npy for the next group. A companion .genes.txt records selected names;
the loader verifies their order. Output JSON records extraction arguments and the feature hash.
Existing features are not overwritten. Extraction with both upstream pretrained weight sets has
not been executed in this environment.
