# Changelog

## Unreleased

- Align the new five-gene training and inference path with the final 46-page manuscript and
  49-page supplement: fixed diffusion variance, three conditional experts, four Table S9
  losses, name-aligned references and sparse gene memory.
- Add native-resolution morphology scoring from existing cell labels and native Prov-GigaPath
  tile export. Require real per-sample metadata, pinned frozen BioBERT v1.1 and current-gene
  co-expression row names. Use all nonempty cellular tiles at inference after the training curriculum.
- Add patient split validation, frozen-encoder LoRA fine-tuning and model plus optimizer resume.
- Train a new five-gene execution-demo checkpoint for 100 updates from random initialization;
  its proxy conditions and single-specimen spatial split are recorded in checkpoint provenance.
- Compute formal PCC per gene and retain undefined values for constant channels; make DDPM and DDIM
  sampler choices explicit.
- Preserve the prior one-gene demo history below as an earlier release, not the current method.

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
