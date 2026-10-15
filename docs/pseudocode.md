# Algorithm outline

The outline below follows the training and sampling code in this repository.

## Training

```text
Read the training configuration and select a one-gene block.
Load the target ST grid, low-resolution ST, registered H&E at both scales,
  masks, reference expression, gene-context prior and encoder/metadata features.
Construct the conditional U-Net and the 1,000-step diffusion process.
For each optimizer update:
  Draw a diffusion timestep and Gaussian noise.
  Noise the target ST and attach the task-selection ratio.
  Run the model with the noisy ST and the conditioning arrays.
  Predict one noise channel, one variance channel and a scale factor.
  Combine the implemented diffusion, alignment, reference, spatial and scale losses.
  Backpropagate and update the optimizer; save checkpoints at the requested interval.
```

The implementation is in `train.py`, `train_part.py`, `guided_diffusion/train_util.py` and
`GaussianDiffusion.training_losses`. The current iteration count is
`int(99 * epoch / batch_size) + 1`, with an inclusive `0..total_iterations` loop.
The scale head uses BatchNorm, so training and microbatch sizes must be at least two.

## Conditional model

`UNetModel.forward` extracts features from noisy ST, low-resolution ST and H&E. It applies
reference fusion through `OTBasedScRNASTFusion`, gene-context memory through
`SpatialGeneMemoryBank`, regional weighting through `HEGuidedRegionalAttention`, and the
cell-fidelity term through `CellFidelityConstraining`. Shared and distinct modality features
are combined with gene-index and metadata features. `CrossTaskCoordinator` produces a latent
update before the decoder predicts noise and variance. See the root README's module map for names.

## Demo inference

```text
Load the one-gene checkpoint in evaluation mode; set the random seed to zero.
For each H&E PNG and same-name NPZ:
  Validate input dimensions and finite values.
  Start a (1, 1, 256, 256) noisy expression tensor.
  Run the complete four-step respaced reverse-diffusion loop.
  Require a finite (1, 1, 256, 256) prediction.
  Save its scalar plane in NPZ and render that plane as a grayscale PNG.
```

The checkpoint and input conditions used for the demo are described in the root README.
