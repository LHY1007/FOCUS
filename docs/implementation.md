# Model implementation and validation

FOCUS implements conditional diffusion for five-gene spatial expression maps. The
model combines registered H&E, high- and low-resolution spatial transcriptomics,
cell geometry, a single-cell reference and pretrained image/expression features.
[Method equations and configuration choices](reproduction.md), [input preparation](real_data.md)
and [demo use](usage.md) provide the corresponding detail.

## Data and conditioning

`Data_all._load_data` selects each gene group using `gene_order.npy` and its ordered
`gene_name.txt`, then retrieves the same named genes from the scRNA reference.
The bundled Xenium tile uses Ttn, Car3, Des, Cav1 and Myoz2, at original panel
columns 365, 52, 116, 53 and 234. The 256-cell P0 muscle reference retains source
cell IDs and donor labels. Spatial co-expression rows are also selected by gene
name; `coexpression_matrix.genes.txt` must match its matrix order. These checks
apply before model execution (Methods 1.3; Supplementary Sections 6–7).

Official 10x nuclear polygons and the native 0.5 µm/pixel H&E field produce a
16×16 complexity grid through `read_xenium_boundaries.py`, `score_complexity.py`
and `complexity_from_native_labels`. Its descriptors include density, nuclear
size, eccentricity/solidity/circularity and hematoxylin texture. Official cell
polygons separately provide 68 selected occupancy planes for the bundled tile.
This subset omits most visible cells. Shape and texture therefore come from the
native nuclear labels, rather than the 10 µm occupancy image (Methods 4,
pp. 39–41). The complexity score's z-score reference population is defined here
as nuclei in the tile because the manuscript does not specify one.

The H&E pathway reads official Prov-GigaPath tile-encoder vectors with shape
`(257, 1536)`; the ST pathway reads scGPT-spatial vectors with shape
`(512, 26, 26)`. Feature export checks the pretrained checkpoint tensors and
gene mapping. The mouse-pup scGPT export matched all 153 required encoder
tensors after a documented non-Flash compatibility patch and an explicit
mouse-to-human ortholog table. Per-sample platform, species, age and tissue text,
with the current five gene names, is encoded by frozen BioBERT v1.1. The
registered `meta_processor` is saved with the model. Exact model versions,
weight hashes and access terms are in [real_data.md](real_data.md).

## Diffusion model

The active path is `train.py` → `train_part.train_fun` → `TrainLoop` →
`GaussianDiffusion.training_losses` → `UNetModel._forward_manuscript`.
`sample.py` and `demos/run.py` use the same U-Net architecture. The sensitivity,
alignment and cell-fidelity activation heads each score their current input;
only scores greater than 0.5 run the corresponding expert. The coordinator
cross-attends over active task tokens and applies FiLM to their features. The
shared U-Net determines output resolution, so activating a task does not add an
independent output head (Methods 1.6, pp. 35–36). Mixed-activation checks verify
that disabled experts are skipped for the affected samples, while activation
heads and coordinator retain training gradients.

The sensitivity branch projects the name-aligned scRNA reference by Sinkhorn
transport into ST features. `HEGuidedRegionalAttention` takes registered RGB
H&E to form its gradient saliency, and `SpatialGeneMemoryBank` applies sparse
top-K attention to the current gene rows. At sparsity 0.15, a 20-row case selects
three entries; the five-row bundled group selects one after integer rounding
(Methods 1.3; Supplementary Sections 7.1–7.2).

`MorphologyAlignment` predicts a deformation field, samples ST features on the
deformed grid, and evaluates NGF, GMC and GMSC with boundary weighting. The
cell-fidelity branch uses native complexity, H&E/ST cell embeddings and their
Pearson-correlation matrices. `cell_consistency` compares the matrices with
`(1 - cosine)/2`; gradients reach both modality features without detaching the
loss path (Methods 1.4–1.5, pp. 33–35). Analytic PCC checks match NumPy,
and gradient checks exercise both paths. The real-condition training log contains
nonzero structure and cell-consistency terms.

`GaussianDiffusion.training_losses` maps Charbonnier, structural,
disentanglement and cell-consistency terms to the Table S9 weights
`0.15/0.20/0.10/0.10` (Methods 3.1, p. 38; Supplementary Table S9, p. 23).
The bundled training run recorded all four terms through update 100; at its last
update, `charb=.316`, `struct=.00856`, `dis=-2`, and `consis=7.22e-6`.
The disentanglement term is signed, so the total need not be positive.

The model predicts five noise channels with fixed diffusion variance. Its
bundled checkpoint strictly matches 352 state keys, including an output
weight of shape `(5,128,3,3)`. Forward diffusion has 1,000 steps. Formal
sampling defaults to 1,000 DDPM steps; the demo uses 1,000 DDIM steps and
labels any shorter `--steps N` run as an execution check (Methods 1.1 and 3.1).
The full real-condition demo produced finite `(5,256,256)` output in 151.90 s;
the five-panel PNG exactly matches the NPZ display transform.

## Training, adaptation and scope

The supplied formal configuration is five genes per group, 2,000 epochs,
batch size four and learning rate `1e-4` (Methods 3.1). `guided_diffusion.splits`
validates patient separation before patch enumeration; `Data_all` records only
valid patch keys and retains group offsets. The bundled sample is a single
**training** tile, not a patient-level test. Historical patient lists and
figure-specific inputs are unavailable.

`meta_processor`, activation heads and coordinator are model modules saved in
checkpoints. Training checkpoints include optimizer state and configuration.
`finetune.py` freezes U-Net input encoders and time embeddings, applies rank-8
LoRA to task matrices by default, and exports merged ordinary weights. Rank,
initialization and several routing details are implementation choices rather
than recovered settings. Resuming does not reproduce data-loader order bit for
bit. The platform preprocessing scripts handle panel metadata and skip empty
Xenium patches. A complete raw-slide preprocessing replay has not been run for
this version.

The demo checkpoint started from random initialization and received 100 optimizer
updates on one public mouse-pup training tile, batch size one and learning rate
`1e-4`. Its co-expression prior also derives from that tile. It has no held-out
accuracy estimate; its output is visibly speckled and should not be treated as
a validated expression reconstruction. The supplied software tests cover data
mapping, conditioning, gradients, adaptation, sampling, package builds and
installation. Full 2,000-epoch training, the manuscript's quantitative figures,
third-party fresh installation and the separate mask/co-expression generators
have not been independently reproduced here. `downstream/` is maintained
separately; its trajectory documentation and annotation comparisons have not
been validated as part of this model release.
