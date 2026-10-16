# Changelog

## Unreleased

- Include the Ttn checkpoint in `demos/` and load it without a separate download.
- Update the Ttn demo checkpoint, input examples and comparison figure.
- Verify all ten demo inputs on a Windows RTX 4060 laptop and record the runtime.
- Add Windows installation timings and package version records.
- Add preprocessing and downstream dependency groups, including native image libraries.
- Read Visium scalefactors with Python's built-in JSON parser.

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
