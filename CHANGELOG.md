# Changelog

## 0.3.4 - 2026-10-20

- Refine installation, demo, data preparation and checkpoint usage documentation.

## 0.3.3 - 2026-10-20

- Include a validated one-sample manifest with the five-gene example and
  document how to construct named tissue co-expression priors from ST grids.
- Include the H&E cell-mask helper in the wheel and align its QC preview grid
  with the saved model mask.
- Update installation, data inputs and distributed batch descriptions; keep
  package verification in GitHub Actions.

## 0.3.2 - 2026-10-18

- Add bounded, parameterized extraction of a registered Xenium tile from
  original HTTP range or local H&E and transcript files, preserving the full
  panel order and all three spatial count grids.
- Support torchrun training with per-rank patch sharding, synchronized
  conditional experts, rank-specific logs, and rank-zero model and optimizer
  checkpoints. The configured batch size is per worker.
- Extend paired five-gene evaluation with H&E gradient alignment metrics,
  RMSE by native-complexity tercile, count-conserving interpolation of the
  observed low-resolution input and matched prediction comparisons.
- Check a previously trained sample's physical ST resolution against its
  checkpoint when sampling it again.

## 0.3.1 - 2026-10-17

- Use measured pre-degradation ST resolution to enable a shared-backbone
  denoiser residual only for source bins coarser than 10 µm. The bundled
  10 µm training tile retains the same inference path and checkpoint tensors.
- Record the routing choice and its tested scope in the model guide.

## 0.3.0 - 2026-10-17

- Replace the default proxy-conditioned ten-pair demonstration with one traced,
  registered 10x mouse-pup H&E/ST tile and genuine official cell and nuclear polygons,
  native morphology, independent P0 muscle scRNA, and checkpoint-verified scGPT-spatial,
  Prov-GigaPath and BioBERT features. Add exact public entrances and a strict
  formal-data loading path. This is a training-tile execution check.
- Add five-gene training and inference with fixed diffusion variance, three
  conditional experts, four weighted losses, name-aligned references and sparse
  gene memory, as described in the Methods and Supplementary Table S9.
- Add native-resolution morphology scoring from existing nuclear labels and native Prov-GigaPath
  tile export. Require real per-sample metadata, pinned frozen BioBERT v1.1 and current-gene
  co-expression row names. Use all nonempty cellular tiles at inference after the training curriculum.
- Add patient split validation, frozen-encoder LoRA fine-tuning and model plus optimizer resume.
- Train the bundled five-gene checkpoint for 100 updates from random initialization
  on the real-condition training tile; provenance records the single training donor.
- Compute formal PCC per gene and retain undefined values for constant channels; make DDPM and DDIM
  sampler choices explicit. Add per-prediction physical-source records and a paired five-gene metrics exporter.
- Read image, ST, mask and encoder arrays on demand; pad variable cell counts in batches. Use
  spatial capture, alignment and cell-mixing diagnostics with measured per-sample ST resolution.
- The earlier single-gene examples remain documented below as archived versions.

## 0.2.0 - 2026-10-16

- Rename `demo/` to `demos/`, retaining ten paired H&E PNG and single-gene NPZ inputs.
- Export a PNG preview and a compressed float32 NPZ prediction for every input; remove TIFF output.
- Add dependency, packaging, contribution, documentation, container and CI files.
- Use the one-gene model with the two-update demo checkpoint.

## One-gene demo - 2026-10-16

- Set supplied training and inference configurations to one generated gene channel.
- Run two optimizer updates and inference on ten real Xenium mouse-pup patches on CB4.
- Publish the execution checkpoint as `demo-1gene-v1` (commit `8333126`).

## Initial repository preparation - 2026-10-15

- Add the MIT license and explicit README license notice.
- Repair configurable paths, tissue selection and the documented feature shapes.
- Validate a five-gene execution example; its checkpoints do not match the one-gene model.
