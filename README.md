# FOCUS: A Foundational Generative Model for Cross-platform Unified Enhancement of Spatial Transcriptomics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

FOCUS (Foundational Generative Model for Cross-platform Unified Enhancement of Spatial Transcriptomics) is a diffusion-based deep learning framework designed to enhance spatial transcriptomics (ST) data resolution across multiple platforms. By integrating multi-modal information including low-resolution ST data, histological images (H&E), single-cell RNA sequencing (scRNA-seq), and gene co-expression networks, FOCUS achieves high-fidelity super-resolution reconstruction of spatial gene expression patterns.

### Key Features

- **Multi-Modal Integration**: Seamlessly combines ST data, H&E images, scRNA-seq references, and gene relationship networks
- **10× Super-Resolution**: Enhances spatial resolution from spot-level to near single-cell resolution
- **Cross-Platform Compatibility**: Supports multiple ST platforms including Xenium, Visium, and others
- **Biologically Informed**: Incorporates gene co-expression networks and scGPT embeddings for biologically meaningful enhancement
- **Advanced Architecture**: Features include:
  - Optimal transport-based alignment between scRNA-seq and ST data
  - H&E-guided spatial attention mechanisms
  - Gene memory networks for co-expression modeling
  - Lightweight module coordinator for multi-scale feature fusion

## Architecture

FOCUS employs a sophisticated U-Net-based diffusion model with several innovative components:

1. **Multi-Modal Encoders**: Separate pathways for ST data, H&E images (multiple scales), and auxiliary features
2. **OT Alignment Module**: Optimal transport for sc→ST feature alignment
3. **H&E Guided Attention**: Multi-scale gradient-based saliency for tissue structure awareness
4. **Gene Memory Network**: Leverages gene co-expression matrices for biological consistency
5. **Constrained Refinement**: Cell-level correlation constraints using cell segmentation masks
6. **Lightweight Module Coordinator**: Token-based negotiation between feature streams

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
@article {Wang2025.12.23.696267,
	author = {Wang, Xiaofei and Liu, Hanyu and Que, Ningfeng and Tao, Chenyang and Jiang, Yu and Jiang, Yixuan and Zhu, Pinan and Zhu, Junze and Li, Xiaoyang and Price, Stephen and Xu, Jianguo and Xi, Jianzhong and Wang, Xinjie and Li, Chao},
	title = {A Foundational Generative Model for Cross-platform Unified Enhancement of Spatial Transcriptomics},
	elocation-id = {2025.12.23.696267},
	year = {2025},
	doi = {10.64898/2025.12.23.696267},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
