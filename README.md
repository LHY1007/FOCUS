# Coordinated Enhancement of Molecular and Spatial Fidelity for Biological Inference in Spatial Transcriptomics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Understanding tissue organization requires accurate molecular quantification and faithful spatial localization. Spatial omics, particularly spatial transcriptomics (ST), face measurement trade-offs that limit simultaneous achievement of molecular and spatial fidelity across spatial resolution, detection sensitivity, molecular-tissue alignment and cell-level fidelity. Existing computational methods typically target individual aspects and optimize them in isolation, despite their interdependence. Here we present FOCUS, a multimodal generative foundation model for coordinated ST enhancement. FOCUS integrates native ST with histology images, cell masks, scRNA-seq references and spatial co-expression priors through task-specific modules, with cross-task coordination enabling joint correction of co-occurring limitations. FOCUS improved the performance of individual enhancement tasks and provided additional gains with coordinated mitigation of coupled limitations. Across immunology, tissue development and oncology, FOCUS recovered biologically consistent patterns obscured in native ST or isolated enhancement, enabling clearer inference of neutrophil-tumor interactions, spatial osteocyte differentiation and primary-metastatic HNSCC programs. Together, FOCUS establishes coordinated enhancement of molecular and spatial fidelity as a strategy for downstream biological inference from ST.

## Installation

### Environment Setup

Create a conda environment using the provided configuration:

```bash
conda env create -f DDPM310.yaml
conda activate ddpm-nm
```

### Requirements

- Python ≥ 3.10
- PyTorch 2.9.0 (CUDA 12.6)
- Key dependencies:
  - `scanpy` (1.11.5) - ST data processing
  - `anndata` (0.11.4) - Data structures
  - `pot` (0.9.6) - Optimal transport
  - `transformers` (4.57.1) - scGPT integration
  - `timm` (1.0.22) - Vision models
  - `mpi4py` (3.1.3) - Distributed training

All dependencies are specified in `DDPM310.yaml`.

## Data Preparation

### Directory Structure

Organize your data following this structure:

```
data_root/
├── {tissue_type}/
│   ├── {sample_id}/
│   │   ├── HR_ST/
│   │   │   └── extract/{layer}/{patch_id}/
│   │   │       └── HR_ST_256.npz
│   │   ├── spot_ST/
│   │   │   └── extract/{layer}/{patch_id}/
│   │   │       ├── spot_ST.npz
│   │   │       └── scgpt_data/
│   │   ├── WSI/
│   │   │   └── extract/{layer}/{patch_id}/
│   │   │       ├── 5120_to256.npy
│   │   │       ├── 320_to16.npy
│   │   │       ├── cell_mask.npy
│   │   │       └── pre_he.npy
│   │   ├── gene_order.npy
│   │   └── gene_name.txt
│   ├── sc/
│   │   └── {scRNA_data}.h5ad
│   └── coexpression_matrix.npy
```

### Required Files

1. **HR_ST_256.npz**: High-resolution ST data (256×256, sparse format)
2. **spot_ST.npz**: Low-resolution spot data (26×26, sparse format)
3. **5120_to256.npy**: H&E patch at 20× magnification
4. **320_to16.npy**: Multi-scale H&E patches (16×16 grid)
5. **cell_mask.npy**: Cell segmentation masks
6. **pre_he.npy**: Pre-extracted H&E features (512-dim)
7. **scgpt_data/**: scGPT embeddings per gene group
8. **{scRNA_data}.h5ad**: Single-cell reference (AnnData format)
9. **coexpression_matrix.npy**: Gene-gene co-expression matrix
10. **gene_order.npy**: Gene indices for the dataset
11. **gene_name.txt**: Corresponding gene names

### Data Preprocessing

Data should be log-normalized with gene-wise scaling. For details on data preparation pipelines, please refer to:

- [BioBERT](https://github.com/dmis-lab/biobert) - Gene name embeddings
- [scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial) - Gene expression embeddings
- [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath) - H&E feature extraction

## Usage

### Training

Configure training parameters in `config/config_train.yaml`:

```yaml
# Key parameters
gene_num: 5         # Genes per training group
batch_size: 8        # Training batch size
SR_times: 10         # Super-resolution factor
epoch: 3000          # Training epochs
lr: 0.0001           # Learning rate
diffusion_steps: 1000
data_root: '/path/to/data/'
log_dir: 'logs/'
```

Launch training:

```bash
python train.py
```

The training script automatically:
- Splits genes into groups of size `gene_num`
- Trains separate models for each gene group
- Saves checkpoints every 2000 steps
- Logs training metrics and loss curves

### Inference

Configure test parameters in `config/config_test.yaml` and run:

```bash
python sample.py
```

Outputs are saved to `TEST_Result-demo/`:
- Predicted gene expression maps (PNG)
- Ground truth comparisons
- Quantitative metrics (CSV): RMSE, SSIM, PCC

### Pre-trained Models

Download pre-trained models from [Dropbox](https://www.dropbox.com/scl/fo/gte7fbz2y14syitka3mb0/AOhsVHx4Rdlk9BU2oJRySv4?rlkey=aytpdtg1ae05jf8i139a2e15c&st=5qi9943t&dl=0).

## Evaluation Metrics

FOCUS is evaluated using:

- **RMSE**: Root mean squared error for intensity accuracy
- **SSIM**: Structural similarity index for spatial patterns
- **PCC**: Pearson correlation coefficient for gene-wise concordance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use gradient checkpointing: `use_checkpoint: true`
   - Enable mixed precision: `use_fp16: true`

2. **Missing Gene Files**
   - Ensure `gene_order.npy` and `gene_name.txt` exist for each sample
   - Verify gene indices match across ST and scRNA-seq data

3. **Slow Training**
   - Adjust `num_workers` in DataLoader
   - Use SSD storage for data
   - Enable `persistent_workers: true`

4. **NaN Losses**
   - Check input data normalization
   - Verify no inf/nan values in co-expression matrices
   - Reduce learning rate

## Citation

If you use FOCUS in your research, please cite:

```bibtex
@article {Wang2026Focus,
	author = {Wang, Xiaofei and Que, Ningfeng and Liu, Hanyu and Tao, Chenyang and Jiang, Yu and Jiang, Yixuan and Zhu, Pinan and Xu, Jianguo and Yan, Huachao and Ye, Buqing and Price, Stephen and Xi, Jianzhong Jeff and Wang, Xinjie and Li, Chao},
	title = {Coordinated Enhancement of Molecular and Spatial Fidelity for Biological Inference in Spatial Transcriptomics},
	elocation-id = {2025.12.23.696267},
	year = {2026},
	doi = {10.64898/2025.12.23.696267},
}
```

## Acknowledgments

FOCUS builds upon several excellent open-source projects:

- **Diffusion Models**: [Improved DDPM](https://github.com/openai/improved-diffusion)
- **BioBERT**: [dmis-lab/biobert](https://github.com/dmis-lab/biobert)
- **scGPT**: [bowang-lab/scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial)
- **GigaPath**: [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath)

## Contact

For questions and feedback:
- Open an issue on GitHub
- Email: xw405@cam.ac.uk

---

**Note**: This is research software. While we strive for correctness, please validate results for your specific application. Contributions and feedback are welcome!
