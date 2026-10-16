# Algorithm outline

This outline describes the revised implementation. The historical training run and its weights are not included.

## Five-gene training

```text
Read a patient-level split and select five target genes by name and offset.
Load paired ST and H&E, cell masks, a matched scRNA reference, the supplied
  co-expression matrix, authentic pretrained features and BioBERT metadata.
For each epoch, visit every selected patch in batches of four.
For each update:
  Sample a timestep from the 1,000-step forward schedule and add Gaussian noise.
  Use three input-derived heads to score sensitivity, alignment and cell fidelity.
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

The default configuration specifies 2,000 epochs. `max_steps` deliberately limits a new execution check. The variance is fixed, so the U-Net emits five channels. See [reproduction.md](reproduction.md) for the equations and newly chosen implementation details.

## Inference

```text
Read the five-gene checkpoint and matching conditions in evaluation mode.
Start from Gaussian noise with shape [batch,5,256,256].
Run 1,000 reverse diffusion steps by default.
Validate finite five-channel output and preserve the selected gene order.
```

The bundled software demo reads ten PNG/NPZ pairs and writes ten five-channel NPZ predictions and ten five-panel PNG previews. `python demos/run.py --steps N` requests a shorter run explicitly.
