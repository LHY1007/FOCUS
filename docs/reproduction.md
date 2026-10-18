# Method implementation and reproduction

This guide maps the Methods and Supplementary Methods to the executable model
and records implementation choices where the published method leaves a value
open. The bundled demo checkpoint belongs to this software version; historical
training records and figure checkpoints are not available here.

## Equations and live call chain

`train.py -> train_part.train_fun -> TrainLoop -> GaussianDiffusion.training_losses`
uses `UNetModel._forward_manuscript`. The same model is used by `sample.py` and `demos/run.py`.

| Manuscript location | Live implementation |
| --- | --- |
| Methods 1.1: five genes; fixed variance | YAML configurations, UNetModel.out, create_gaussian_diffusion |
| Methods 1.3: uniform-marginal Sinkhorn and squared Euclidean cost | manuscript.reference_projection; reference columns matched by gene name in Data_all |
| Supplementary 7.1: gene-row memory and sparse selection | Data_all._coexpression_features selects current rows by sidecar gene names; SpatialGeneMemoryBank and SparseGeneMemoryAttention retain top-K at ratio 0.15 |
| Supplementary 7.2: H&E gradient saliency | HEGuidedRegionalAttention, called with registered RGB H&E |
| Methods 1.4: local modulation and shared/unique features | MorphologyAlignment.local_kernels, shared, unique |
| Methods 1.4: deformation, NGF, GMC, GMSC and boundary weights | MorphologyAlignment.forward, morphology.boundary_weights |
| Methods 1.5: cellular selection and cross-attention | complexity_grid, ManuscriptConditioning.cellular_fusion |
| Methods 1.5, p. 35: PCC and (1-cosine)/2 | pearson_cells, cell_consistency; both encoder paths retain gradients |
| Methods 1.6: three heads, threshold >0.5, active tokens and FiLM | ManuscriptConditioning.task_scores, ActiveTaskCoordinator; inactive experts skipped per sample |
| Methods 1.6: physical resolution gate | `st_resolution.json` → `Data_all` → `resolution_gate`; measured pre-degradation ST bin size >10 µm activates the shared backbone gate |
| Methods 2: patient separation before patch extraction | guided_diffusion.splits; loader filters samples and rejects cross-split patients |
| Methods 3.1, Table S9: four-term loss | 0.15 charb + 0.20 struct + 0.10 dis + 0.10 consis |
| Methods 3.2: frozen encoders, LoRA and prompt routing | configure_finetuning, LowRankUpdate, ActiveTaskCoordinator.router |
| Methods 4, pp. 39-41: density plus morphology heterogeneity | score_complexity -> morphology.complexity_from_native_labels -> Data_all._load_data -> UNetModel._forward_manuscript; radius 3 um, balance 0.5, equal component weights. The bundled demo uses native H&E and official 10x nuclear labels for this calculation. |

Charbonnier uses `sqrt((epsilon_prediction-epsilon)^2+gamma^2)`, gamma=1e-4.
Disentanglement uses `-cos(shared_HE,shared_ST)+cos(unique_HE,unique_ST)`.
The total has no additional learned-variance, scale, OT-reconstruction, attention-weighted
noise or shared/distinct ratio losses.

## Explicit choices where the manuscript is incomplete

These values define the new implementation; they are not recovered historical settings.

- Sinkhorn regularization 10, 100 log-domain iterations, differentiable transport and one reference per sample.
- A measured source ST bin coarser than 10 µm adds a 0.1-scale denoiser-feature
  residual to the shared backbone before its middle block; no extra output head
  is introduced. At 10 µm or finer, the residual is zero, so the bundled
  checkpoint follows its original trained path. This routing gain was not
  provided by the paper, and no real coarse-bin cohort has been evaluated.
- Activation heads receive capture sparsity, ST/H&E edge mismatch and between-cell
  expression mixing diagnostics. Their initial feature thresholds are 0.05, 0.40
  and 0.25, respectively. These are software calibration choices, not historical
  task labels. A straight-through gate carries training gradients for active heads
  while forward expert selection uses the stated hard threshold.
- Cross-modulation uses 3x3 kernels. Deformation uses two convolutions, maximum normalized displacement
  0.1. Shared-feature perturbation standard deviation is 0.01; numerical c is 1e-8.
- The unique-feature cosine coefficient is 1.0; the paper specifies its sign but no value.
- Descriptor deviations are mean absolute z scores relative to cells in the input tile. Size uses log
  area and equivalent diameter; texture uses hematoxylin entropy and variance. Cells are assigned by
  centroid. Full training reads `complexity_grid.npy` scored from registered native H&E and existing
  nuclear-instance labels at 0.5 um/pixel, before the 10 um/pixel model input loses subcellular detail.
  The demo uses official nuclear polygons for complexity and official cell polygons for
  the separate 68-plane occupancy input; that occupancy is a selected subset of visible cells.
- Methods 1.5 says both "early iterations focus on simple patches" and "only patches with c above
  the current threshold" as that threshold increases. These statements conflict. This implementation
  follows the simple-to-complex order, retaining the lowest-scoring 10% of nonempty tiles initially
  and growing the set to 100%. Evaluation always uses all nonempty tiles. The original schedule was
  not supplied.
- Reference sampling is without replacement, seed 0, default 256 cells per tissue, after zero-count-cell
  removal. Normalization uses 10,000 followed by log1p. Name-based selection follows normalization;
  no initial 100-gene truncation occurs.
- The paper names BioBERT without a version. This implementation pins official
  `dmis-lab/biobert-v1.1` revision `551ca18efd7f052c8dfa0b01c94c2a8e68bc5488` and validates
  the model file hash. Each sample supplies its own platform, species and tissue text; the selected
  five genes are prefixed before frozen encoding.
- Prov-GigaPath supplies one tissue vector and 256 cellular vectors from the registered native
  0.5 um/pixel H&E field, with 320-pixel cellular tiles associated with their actual grid positions.
  A learned linear map matches the native dimension. scGPT-spatial spot vectors return to their
  row-major 26x26 grid. Neither is reshaped into an arbitrary RGB image.
- Co-expression rows are selected by name and pooled to 64 values per current gene before learned
  memory projections. This fixed-width pooling is a new choice; the paper does not give the
  projection architecture. A matching row-name sidecar is required for full training.
- LoRA rank defaults to 8 with a zero-initialized update. Frozen task matrices include convolution and
  attention weights. All U-Net input encoding paths and the time embedding are frozen; decoder and
  matching/fusion layers remain trainable. Resume loads model and optimizer states and counts completed
  updates. Data-loader ordering after interruption is not bitwise reproduced.

## Commands

Prepare authentic inputs using [usage.md](usage.md), then run:

```bash
python -m guided_diffusion.splits data/samples.csv
python train.py --config config/config_train.yaml --data_root data/prepared --split_manifest data/samples.csv
python sample.py --config config/config_test.yaml --data_root data/prepared --split_manifest data/samples.csv
python train.py --config config/config_finetune.yaml --init_checkpoint models/foundation.pt --split finetune
```

For multiple visible GPUs, use `torchrun --nproc_per_node=N train.py` with the
same configuration arguments. The distributed sampler shards patch indices per
epoch and repeats a few indices if the patch count is not divisible by the
worker count;
gradients are synchronized even when workers activate different experts.
Each worker records its own losses, while rank zero saves the shared model,
optimizer and configuration. `batch_size` is per worker. The effective global
batch is `N × batch_size`; choose it deliberately when comparing experiments.
The source patient manifest remains the split authority. Resume restores model
and optimizer state and resumes the deterministic epoch shard order, but does
not reproduce the exact random noise sequence from an interrupted run.
When sampling a specimen recorded in the training configuration, its measured
ST resolution and source description must still match the checkpoint record;
new specimens supply their own measured values.

Training uses five genes, 2,000 epochs, batch size 4, AdamW learning rate 1e-4 and 1,000 diffusion steps.
Formal sampling uses 1,000 DDPM reverse steps by default; the bundled execution demo uses 1,000 DDIM steps.
The actual configuration is saved beside weights.
`max_steps` optionally limits execution checks; zero retains the full epoch count.

The registered Xenium mouse-pup tile can be extracted from the official raw
H&E and transcript HTTP range sources with the command in [real_data.md](real_data.md).
For tile `3_10`, the raw replay produced exactly the same 399-channel
`HR_ST_256`, `HR_ST_128` and `spot_ST` matrices as the separately prepared
tile: all three contain 10,590,521 panel transcripts. The 256-pixel H&E
thumbnail differs by at most one 8-bit level between the OpenCV and libvips
bilinear resamplers. The replayed tile can be placed into the formal tree
without changing its existing cell labels, independent reference, co-expression
prior or pretrained features.
That formal tree passed strict loading and a two-step DDPM inference with the
100-update real-condition checkpoint, producing a finite five-channel output.
The full 1,000-step demonstration was previously run on the matching
registered tile, whose H&E was resampled by libvips.

The patient utility creates or validates real assignments. Original patient lists, figure-specific weights
and a figure-to-input command map were not supplied. The bundled five-gene demo weight was trained from random initialization for 100 updates
on one public mouse-pup training tile, batch 1. It uses a P0 muscle scRNA reference
from an independent study, official 10x cell polygons, checkpoint-verified scGPT-spatial and
Prov-GigaPath features, and a same-tile co-expression prior. There is no held-out
validation or test donor. See [real_data.md](real_data.md) and [usage.md](usage.md).

## Verification scope

Sixteen targeted synthetic tests cover gene matching, offsets, skipped indices, current co-expression
rows, per-sample gene prompts, native morphology,
PCC and cell gradients,
task disabling, training/evaluation cellular selection, deformation gradients, patient separation and
LoRA initialization. A CB4 GPU check
uses explicitly synthetic inputs to verify five-gene forward/backward execution, metadata/task
gradients, encoder freezing, a LoRA optimizer step, merged-weight strict loading and reverse sampling.

A full 2,000-epoch experiment and Fig. 2-7 quantitative results have not been rerun.
Both official pretrained encoder weight sets were loaded and real features were exported for
the bundled registered tile. Strict formal loading and 100 real-condition optimizer updates
completed. The genuine-condition 1,000-step DDIM run took 148.02 s on an
RTX 6000 Ada (0.43 GiB peak allocated, 0.70 GiB peak reserved) and its
five-panel PNG exactly matches its NPZ. The preview remains visibly speckled;
this is a training-tile execution check, not independent accuracy evidence.
The formal `sample.py` path also passed an explicitly shortened two-step DDPM
execution on that same train tile. Two-rank CPU/Gloo checks exercised patch
sharding, expert-conditional gradient synchronization, checkpoint and optimizer
resume, and one complete real-condition forward/backward update. A separate
two-rank real-condition LoRA run updated the adapted model, resumed its model
and optimizer for a second update, and exported strictly loadable merged
five-channel weights. These are execution checks on the single training tile.
The seven-V100
historical run was not repeated. Package 0.3.2 source and wheel archives
built and passed `twine check`; the wheel installed into a separate target
directory and exposed the paired evaluator and Xenium replay command. The
bundled model and its previously validated 1,000-step execution are unchanged.
