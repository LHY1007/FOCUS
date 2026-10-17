# Independent audit against the final manuscript and original review

This audit uses the supplied 46-page final main PDF and 49-page supplement as the
authority. The shared conversation titled “检查GitHub与论文一致性” was checked against
the complete local 261-line review, not only its two screenshots. The running
implementation is a new manuscript-aligned implementation; historical experiment
code, patient assignments and figure checkpoints were not recovered.

## Ten requested checks

| Item and paper location | Live call chain | Acceptance evidence and limit |
| --- | --- | --- |
| 1. Three task heads, `score > 0.5`, active experts; shared resolution. Main Methods 1.6, pp. 35–36. | `UNetModel._forward_manuscript` → `ManuscriptConditioning.task_scores` → index only active sensitivity/alignment/cell-fidelity branches → `ActiveTaskCoordinator` active tokens and FiLM; shared U-Net emits the high-resolution maps. | `test_disabled_experts_do_not_affect_other_samples` checks mixed activation; the current GPU integration check forbids and confirms skipped inactive modules. Real training logged nonzero structure and consistency terms. The activation heads have nonzero gradients and are included in strict checkpoint loading. |
| 2. Exact scRNA/ST gene correspondence. Main Methods 1.3, pp. 31–32; Supplementary §6, p. 11. | `Data_all._load_data` resolves source `gene_order.npy`/`gene_name.txt` and reference `var_names`; `reference_projection` asserts `[batch,cells,selected_genes]`. `demos/assemble_real_data.py` reads the 399-column 10x panel. | Actual columns for Ttn/Car3/Des/Cav1/Myoz2 are 365/52/116/53/234, and the independent raw-count reference has all five names. HR/LQ count totals match by gene. Offset and wrong-order unit checks pass; neither the first 100 genes nor `sc[:,:C]` is used. |
| 3. Density plus morphology, including size, shape and texture. Main Methods 4, pp. 39–41. | Official 10x **nucleus** polygons → `read_xenium_boundaries.py` → `score_complexity.py` → `complexity_from_native_labels` → formal loader → `cellular_fusion` and `cell_consistency`. Official **cell** polygons separately supply the 68 occupancy planes. | The 16×16 real grid comes from native 0.5 µm/pixel H&E and 46,035 visible vendor nuclei, not grayscale entropy or a 10 µm proxy mask. Shape uses eccentricity/solidity/circularity; size uses log area/diameter; texture uses hematoxylin entropy/variance. The source-hash sidecar records inputs. The paper leaves the z-score reference unspecified; this implementation uses cells in the tile. |
| 4. PCC cell matrices and `(1-cosine)/2` with gradient. Main Methods 1.5, p. 35, equation at line 870. | `UNetModel._forward_manuscript` → `cell_consistency` → per-cell H&E/ST embedding means → `pearson_cells` → vectorized cosine → `GaussianDiffusion.training_losses`. | Analytic unit checks match NumPy PCC and backpropagate to both modality features; no detach is on the loss path. The final real training log has a nonzero `consis` term. Stored detached scalar diagnostics do not enter the loss. |
| 5. Fixed variance, five output channels and strict weights. Main Methods 1.1, pp. 29–30. | `config_train.yaml`/`config_test.yaml` → `sr_create_model_and_diffusion` → `UNetModel.out`; `demos/run.py` strictly loads the five-gene checkpoint. | `learn_sigma=false`, checkpoint has 352 matching state keys and output weight shape `(5,128,3,3)`; model creation reports 1,000 diffusion steps. The actual 1,000-step run produced `(5,256,256)` finite output. Historical one-gene weights are incompatible with this model. |
| 6. Four loss terms with Table S9 weights. Main Methods 3.1, p. 38, lines 948–949; Supplementary Table S9, p. 23. | `GaussianDiffusion.training_losses` in manuscript mode maps `charb`, `struct`, `dis`, `consis` to `0.15/0.20/0.10/0.10`; `train_part` passes the actual configuration. | Real-condition training logged all four terms through update 100. At step 100: `charb=.316`, `struct=.00856`, `dis=-2`, `consis=7.22e-6`; its negative total is allowed by the signed disentanglement term. The historical multi-term branch remains for legacy checkpoints but is not selected. |
| 7. Sparse top-K ratio `0.15`. Supplementary §7.1, pp. 11–12; Table S9 p. 23. | `Data_all._coexpression_features` selects the current named rows → `UNetModel.MemoryNetwork.sparse_attention.sparsity=.15` → `torch.topk` over the active memory rows. | A 20-row fixture selects three entries; the actual five-row group selects one because discrete K is `max(1,floor(.15×5))`. The ratio affects the live attention path; one of five is an effective 20% for this group. |
| 8. Genuine sample metadata, five names, frozen BioBERT. Main Methods 1.1, p. 29, lines 710–713. | Per-sample `metadata_prompt.txt` → `Data_all` prefixes selected gene names → verified official `dmis-lab/biobert-v1.1` → persistent `meta_processor` in the U-Net. | The public tile prompt records Xenium, mouse P1 and a muscle-enriched field, not a fixed brain/cancer prompt. The 768-vector is nonzero; model checkpoint strictly loads the registered processor. BioBERT revision and weight hash are pinned in `real_data.md`. |
| 9. Five genes, 2,000 epochs, batch 4, learning rate 1e-4. Main Methods 3.1, p. 38. | `config/config_train.yaml` → `train.py` → `train_part` → complete-epoch `TrainLoop`; split manifest is checked before patch enumeration. | Formal defaults are 5/2000/4/1e-4. The separate public demo executed 100 updates on one train tile with batch 1 and 1e-4; it is not evidence that 2,000 epochs or paper accuracy were reproduced. |
| 10. Forward and reverse 1,000 steps; demo override explicit. Main Methods 1.1 and 3.1, p. 38. | `create_gaussian_diffusion` uses 1,000 forward timesteps; `sample.py` defaults to 1,000 DDPM reverse steps; `demos/run.py` defaults to 1,000 DDIM steps and labels an optional shorter `--steps N`. | The genuine-condition full DDIM chain completed in 151.90 s. Output is finite `(5,256,256)`, and the 256×1280 PNG exactly equals the rounded NPZ panels. A two-step check was used only to catch execution errors. |

## Remaining points from the full original conversation

- **H&E guidance and alignment:** Supplementary §7.2, p. 12 derives the
  saliency map from H&E gradients. `_forward_manuscript` calls
  `he_attention(enhanced, he[idx])`, with normalized registered RGB H&E.
  Main p. 33 deformation/NGF/GMC/GMSC follows
  `MorphologyAlignment.flow → grid_sample → Sobel → boundary_weights → structural loss`.
  The real run's nonzero `struct` term proves this path contributed to optimization.
- **State and adaptation:** `meta_processor`, three activation heads and the
  coordinator are persistent model modules. `train_part` saves actual config and
  reference selection; model/optimizer checkpoints support resume. `finetune.py`
  freezes U-Net input encoders and time embeddings, parametrizes task matrices
  with rank-8 LoRA by default, and exports merged ordinary weights. Rank and
  initialization are new choices, not recovered historical settings; resumed
  data-loader order is not bitwise identical.
- **Split and skipped patches:** `guided_diffusion.splits` rejects a donor in
  multiple splits. `Data_all` filters by that manifest and records only validated
  patch keys, avoiding reference/index drift after a missing file. The public
  example contains one train donor; there is no independent test. The original
  paper's patient lists and overlapping-field audit files were not supplied.
- **Legacy preprocessing:** Visium now imports standard `json`; Xenium imports
  `shutil` and skips blank patches before reading an unset tile stack. Source
  regression checks pass. The old Visium/Visium HD/Xenium H&E tile sorting by
  Shannon entropy is historical preprocessing and is **not** the live paper
  complexity score. A full raw-slide replay was not run in this delivery.
- **Protected or external items:** The separately owned StarDist2D mask
  generator and co-expression construction were not changed or independently
  validated. The demo's existing prior is derived from its training tile.
  `downstream/trajectory/00_subsample.py` now exists, while
  `downstream/README.md` still says the 50,000-cell generator is absent; its
  route and the RCTD/Tangram versus Methods 7 annotation wording remain
  unverified and untouched. No genuine third-party fresh-install report exists.
- **Research and release scope:** Official Prov-GigaPath is gated; exact
  weights and public entrances are documented. scGPT-spatial needed a pinned
  non-Flash compatibility patch and explicit mouse-to-human ortholog mapping.
  The 10x H&E/ST raw-to-tile extraction was inherited from an existing public
  processed tree, then its real conditions and formal loader were executed here.
  Original historical checkpoints, patient splits, figure-to-input mappings and
  full Fig. 2–7 accuracy remain unavailable. A 0.3.0 source archive and wheel
  built and passed `twine check`; the installed wheel's bundled two-step demo
  ran. The old v0.2.0 prerelease is not evidence for this repaired tree;
  identify the delivered main commit directly until a new immutable release is tagged.
