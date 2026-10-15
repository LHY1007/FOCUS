# Changelog

## 0.2.0 - 2026-10-16

- Rename `demo/` to `demos/`, retaining ten paired H&E PNG and single-gene NPZ inputs.
- Export a PNG preview and a compressed float32 NPZ prediction for every input; remove TIFF output.
- Add dependency, packaging, contribution, documentation, container and CI files.
- Preserve the one-gene model and the execution-only checkpoint; scientific inference quality has
  not been established by this short-run demo.

## One-gene demo - 2026-10-16

- Set supplied training and inference configurations to one generated gene channel.
- Run two optimizer updates and inference on ten real Xenium mouse-pup patches on CB4.
- Publish the execution checkpoint as `demo-1gene-v1` (commit `8333126`).

## Initial repository preparation - 2026-10-15

- Add the MIT license and explicit README license notice.
- Repair configurable paths, tissue selection and the documented feature shapes.
- Validate a five-gene execution example; its checkpoints do not match the one-gene model.
