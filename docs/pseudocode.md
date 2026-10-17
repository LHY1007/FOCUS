# Algorithm outline

This outline summarizes the current implementation. Historical training weights are not included.

## Five-gene training

```text
Read a patient-level split, explicit source ST bin size and five target genes by name and offset.
Load paired ST and H&E, cell masks, a tissue-matched scRNA reference, the supplied
  co-expression matrix, authentic pretrained features and BioBERT metadata.
For each epoch, visit every selected patch in batches of four.
For each update:
  Sample a timestep from the 1,000-step forward schedule and add Gaussian noise.
  Score capture sparsity, ST/H&E spatial edge agreement and between-cell expression
    mixing with three input-derived heads; activate resolution when the measured
    source ST bin is coarser than 10 µm.
  Execute only experts whose score exceeds 0.5; the high-resolution U-Net is shared.
  Project the gene-aligned scRNA reference through Sinkhorn transport when sensitivity
    is active, and select the top 15% of gene-memory entries.
  For cell fidelity, score density and shape, size and texture diversity; gradually
    admit more complex H&E tiles and compare H&E/ST cell PCC matrices.
  For alignment, deform shared ST features toward H&E and evaluate NGF, GMC and GMSC.
  Coordinate active expert tokens with attention and FiLM, then predict five noise maps.
  Minimize 0.15 Charbonnier + 0.20 structure + 0.10 disentangling
    + 0.10 cell-consistency loss; update AdamW at learning rate 1e-4.
  Save model and optimizer states at the requested completed-step interval.
```

The default configuration specifies 2,000 epochs. `max_steps` limits a shorter execution run when set. The variance is fixed, so the U-Net emits five channels. See [reproduction.md](reproduction.md) for equations and implementation details.

## Inference

```text
Read the five-gene checkpoint and matching conditions in evaluation mode.
Start from Gaussian noise with shape [batch,5,256,256].
Run 1,000 reverse diffusion steps by default.
Validate finite five-channel output and preserve the selected gene order.
```

The bundled software demo reads one registered real-condition PNG/NPZ pair from a training tile and writes one five-channel NPZ prediction and one five-panel PNG preview. Its observed ST arrays remain separate from model predictions. `python demos/run.py --steps N` requests a shorter run explicitly.
