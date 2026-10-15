# Reproduction

## Included example

Install FOCUS as described in the root README, then run:

```bash
python demos/run.py
```

The runner fixes the NumPy and PyTorch seeds to zero and uses 16 DDIM sampling steps.
The checkpoint SHA-256 is checked before loading. Ten input pairs produce ten PNGs and ten NPZs
in `demos/output/`. Each NPZ contains a finite float32 `(256, 256)` prediction; the PNG shows
that array using the display rule in [usage.md](usage.md).

The included checkpoint predicts Ttn. Its input conditions are listed in [usage.md](usage.md).
All ten examples passed on CB4 and on Windows 10 with an RTX 4060 Laptop GPU (8 GB).
The Windows run took 30.91 seconds, including startup, model loading, inference and writing
ten PNGs and ten NPZs; checkpoint download time was excluded. All predictions were finite,
single-channel float32 arrays of shape `(256, 256)`, and each PNG matched its NPZ's display
conversion. Tested system and package versions are listed in the root README.

## Paper experiments

The settings below were checked against [preprint version 3, posted 12 October 2026](https://www.biorxiv.org/content/10.64898/2025.12.23.696267v3.full.pdf).
The algorithm is described in Methods sections 1.1–1.6 and 3, on PDF pages 29–39.

| Setting | Paper | Current repository |
| --- | --- | --- |
| Generated genes per iteration | 5, Methods 1.1 footnote | 1 in both supplied configurations |
| Training length | 2,000 epochs, Methods 3.1 | `epoch: 3000`; the loop derives its step count from `int(99 * epoch / batch_size) + 1` |
| Training batch size | 4, Methods 3.1 | 8 in `config/config_train.yaml` |
| Learning rate | 0.0001, Methods 3.1 | 0.0001 |
| Forward diffusion steps | 1,000, Methods 3.1 | 1,000 |
| Reverse diffusion steps | 1,000, Methods 3.1 | 400 in `config/config_test.yaml`; 16 in the demo |
| Data split | Patient-level split before patch extraction, Methods 2 | The loader reads the supplied root; patient split lists are not included |
| Fine-tuning | Frozen encoder and LoRA updates, Methods 3.2 | A separate fine-tuning command is not documented |

For other prepared datasets, set the data, output and model paths in the YAML files and run:

```bash
python train.py --config config/config_train.yaml
python sample.py --config config/config_test.yaml
```

These commands use the current one-gene configuration. Checkpoint dimensions must match it.
The [input layout](../README.md#data-preparation) and [preprocessing steps](../data_preprocessing/README.md)
describe the required files. Analysis-specific inputs and commands are in
[downstream/README.md](../downstream/README.md).

Full paper reproduction still requires the original patient splits, experiment configurations,
matching trained checkpoints and a mapping from each figure to its inputs and commands.
The trajectory scripts also require the 50,000-cell subsampled object described in the downstream
README; the script that creates that object is not included. The current demo cannot substitute
for those experiments.
