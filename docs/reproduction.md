# Method implementation and reproduction

This guide maps the Methods and Supplementary Methods to the executable model
and records the configuration used by the current software.

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

## Configuration choices

- Sinkhorn regularization 10, 100 log-domain iterations, differentiable transport and one reference per sample.
- A measured source ST bin coarser than 10 µm adds a 0.1-scale denoiser-feature
  residual to the shared backbone before its middle block; no extra output head
  is introduced. At 10 µm or finer, the residual is zero, so the bundled
  checkpoint follows its trained 10 µm path.
- Activation heads receive capture sparsity, ST/H&E edge mismatch and between-cell
  expression mixing diagnostics. Their initial feature thresholds are 0.05, 0.40
  and 0.25, respectively. A straight-through gate carries training gradients for active heads
  while forward expert selection uses the stated hard threshold.
- Cross-modulation uses 3x3 kernels. Deformation uses two convolutions, maximum normalized displacement
  0.1. Shared-feature perturbation standard deviation is 0.01; numerical c is 1e-8.
- The unique-feature cosine coefficient is 1.0.
- Descriptor deviations are mean absolute z scores relative to cells in the input tile. Size uses log
  area and equivalent diameter; texture uses hematoxylin entropy and variance. Cells are assigned by
  centroid. Full training reads `complexity_grid.npy` scored from registered native H&E and existing
  nuclear-instance labels at 0.5 um/pixel, before the 10 um/pixel model input loses subcellular detail.
  The demo uses official nuclear polygons for complexity and official cell polygons for
  the separate 68-plane occupancy input; that occupancy is a selected subset of visible cells.
- Training admits the lowest-scoring 10% of nonempty tiles initially and grows
  the set to 100% as complexity increases. Evaluation uses all nonempty tiles.
- Reference sampling is without replacement, seed 0, default 256 cells per tissue, after zero-count-cell
  removal. Normalization uses 10,000 followed by log1p. Name-based selection follows normalization;
  no initial 100-gene truncation occurs.
- Metadata encoding pins official `dmis-lab/biobert-v1.1` revision
  `551ca18efd7f052c8dfa0b01c94c2a8e68bc5488` and validates
  the model file hash. Each sample supplies its own platform, species and tissue text; the selected
  five genes are prefixed before frozen encoding.
- Prov-GigaPath supplies one tissue vector and 256 cellular vectors from the registered native
  0.5 um/pixel H&E field, with 320-pixel cellular tiles associated with their actual grid positions.
  A learned linear map matches the native dimension. scGPT-spatial spot vectors return to their
  row-major 26x26 grid. Neither is reshaped into an arbitrary RGB image.
- Co-expression rows are selected by name and pooled to 64 values per current gene before learned
  memory projections. A matching row-name sidecar is required for training.
- LoRA rank defaults to 8 with a zero-initialized update. Frozen task matrices include convolution and
  attention weights. All U-Net input encoding paths and the time embedding are frozen; decoder and
  matching/fusion layers remain trainable. Resume loads model and optimizer states and counts completed
  updates. Each launch seeds the sampler from the configured seed.

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
and optimizer state and the recorded update count. Distributed runs use that count
and the configured seed to select the epoch shard. Each launch seeds its process
RNG for noise sampling.
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

The patient utility creates or validates sample assignments. The bundled five-gene
demo weight was trained from random initialization for 100 updates
on one public mouse-pup training tile, batch 1. It uses a P0 muscle scRNA reference
from an independent study, official 10x cell polygons, checkpoint-verified scGPT-spatial and
Prov-GigaPath features, and a same-tile co-expression prior. See
[real_data.md](real_data.md) and [usage.md](usage.md) for the input record.

## Runtime checks

Sixteen targeted synthetic tests cover gene matching, offsets, skipped indices, current co-expression
rows, per-sample gene prompts, native morphology,
PCC and cell gradients,
task disabling, training/evaluation cellular selection, deformation gradients, patient separation and
LoRA initialization. A CB4 GPU check
uses explicitly synthetic inputs to verify five-gene forward/backward execution, metadata/task
gradients, encoder freezing, a LoRA optimizer step, merged-weight strict loading and reverse sampling.

Official pretrained encoder weights were loaded to export features for the
bundled tile. Its checkpoint received 100 updates. The 1,000-step DDIM output
is a functional demonstration with matching NPZ and five-panel PNG; the
measured RTX 6000 Ada runtime was 148.02 s. Two-rank CPU/Gloo checks exercised
sharding, expert-conditional synchronization, checkpoint and optimizer resume,
and LoRA adaptation. Package installation and build checks run in GitHub Actions.
