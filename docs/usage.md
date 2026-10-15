# Inputs and outputs

Run `python demos/run.py` from a source checkout with the dependencies installed and a CUDA GPU.
The script reads ten same-name pairs from `demos/input/` and creates `demos/output/` if needed.
The checkpoint is downloaded to `models/` on the first run. Use `--checkpoint /path/to/model.pt`
for offline use or an independently trained checkpoint with the same one-gene architecture.

## One input pair

`01.png` is a 256 x 256 RGB H&E image. `01.npz` contains these arrays:

| Key | Shape before adding the batch dimension | Purpose |
| --- | --- | --- |
| `low_res` | `(1, 26, 26)` | Single-gene low-resolution ST expression |
| `WSI_320` | `(256, 3, 16, 16)` | Multi-scale H&E subpatches |
| `gene_class` | `(1,)` | Target-gene index |
| `Gene_index_map` | `(1, 256, 256)` | Target-gene spatial index map |
| `metadata_feature` | `(768,)` | Precomputed BERT metadata vector |
| `scale_gt` | scalar | Scale field used by the shared model interface |
| `co_expression` | `(10, 10)` | Gene-context prior, separate from the one generated channel |
| `WSI_mask` | `(M, 256, 256)` | Proxy mask components |
| `sc` | `(100, 1)` | Reference expression for the target gene |
| `scgpt` | `(512, 26, 26)` | Encoder-feature placeholder |
| `pre_he` | `(3, 512)` | H&E-feature placeholder |

Text fields identify the gene (`Ttn`), source tile, source URL and limitations. The demo was derived
from public Xenium mouse-pup data and an existing public WMB midbrain reference. It uses proxy
masks, zero scGPT/GigaPath arrays and an unmatched reference.

## One output pair

`01.npz` contains `prediction`, a finite float32 `(256, 256)` array: one generated expression channel.
It also records `display_min=-1` and `display_max=1`. These values are raw diffusion output, not
calibrated transcript counts. `01.png` is a single-channel, 8-bit grayscale preview calculated as:

```python
preview = np.rint((np.clip(prediction, -1, 1) + 1) * 127.5).astype(np.uint8)
```

All ten inputs produce exactly ten PNGs and ten NPZs. The runner writes no metrics, comparison
images, log files or model checkpoints into `demos/output/`. Source packages leave it empty.
The output names match the input names. Move prior results elsewhere before reusing those names.

## Your own data

For full training and evaluation, prepare the structured dataset described in the root README and
`data_preprocessing/README.md`, set `gene_num: 1` and a matching gene order, and run `train.py` or
`sample.py` with `--config`. Align reference genes and expression columns by gene identity.
Register the H&E image to the ST coordinates. Supply computed encoder features and a tissue-matched
single-cell reference.
