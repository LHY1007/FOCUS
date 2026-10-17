"""Run FOCUS inference and compute the evaluation metrics.

Reloads a checkpoint written by `train_part.py`, runs the reverse diffusion
process over the test split and reports the per-gene-group metrics.

    python sample.py
"""

import os, re, glob, gc, json, yaml, torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mpi4py import MPI
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import sr_create_model_and_diffusion, add_dict_to_argparser
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if torch.cuda.is_available():
    torch.cuda.set_device(rank % torch.cuda.device_count())
# Match train.py: initialize the CUDA/runtime stack before importing Scanpy.
from guided_diffusion.load_dataset import load_data, collate_patches
def create_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/config_test.yaml"), help="Path to YAML configuration file")
    config_args, _ = parser.parse_known_args()
    with open(config_args.config, "r") as file:
        config = yaml.safe_load(file)
    config.setdefault("output_dir", "TEST_Result-demo")
    config.setdefault("checkpoint_root", "logs")
    config.setdefault("noise_sigma", 1.0)
    add_dict_to_argparser(parser, config)
    return parser

args = create_argparser().parse_args()
Output = args.output_dir

# ========== 辅助函数 ==========
def noise_like(tensor):
    return torch.randn_like(tensor)

def compute_metrics(gt, pred):
    if gt.shape != pred.shape or gt.ndim != 3 or not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError('Metrics require matching finite [H,W,gene] arrays')
    rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
    ssim_v = float(np.mean([ssim(gt[..., k], pred[..., k], data_range=1.0)
                            for k in range(gt.shape[-1])]))
    correlations = []
    for k in range(gt.shape[-1]):
        a, b = gt[..., k].ravel(), pred[..., k].ravel()
        correlations.append(float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float('nan'))
    # A constant channel has undefined PCC; preserve that fact in metrics.csv.
    return rmse, ssim_v, float(np.mean(correlations))

def save_sample_images(gt, pred, idx, outdir):
    os.makedirs(outdir, exist_ok=True)
    for i in range(gt.shape[-1]):
        plt.imsave(os.path.join(outdir, f"{idx}_{i}gt.png"), gt[..., i], cmap="viridis")
        plt.imsave(os.path.join(outdir, f"{idx}_{i}pred.png"), pred[..., i], cmap="viridis")


def main(Output):
    pre_model_dir = args.pre_model_dir
    model_dirs = sorted(glob.glob(os.path.join(args.checkpoint_root, args.dataset_use, f"{args.SR_times}X", "G*")))

    if len(model_dirs) == 0:
        raise FileNotFoundError(f"No checkpoint directories under {args.checkpoint_root}")

    for model_dir in model_dirs:
        dir_name = os.path.basename(model_dir)
        g_part = dir_name.split("G")[1].split("_")[0]
        start_gene = int(g_part.split("-")[0])

        merged = os.path.join(model_dir, "merged.pt")
        ckpts = glob.glob(os.path.join(model_dir, "model*.pt"))
        if not ckpts and not os.path.isfile(merged):
            continue
        model_path = merged if os.path.isfile(merged) else max(
            ckpts, key=lambda ck: int(re.search(r"model(\d+)\.pt", ck).group(1)))
        saved_config = Path(model_dir) / 'config.yaml'
        if not saved_config.is_file():
            raise FileNotFoundError(f'{saved_config}: required to verify checkpoint architecture and gene range')
        with saved_config.open(encoding='utf-8') as f:
            trained = yaml.safe_load(f)
        for key in ('gene_num', 'num_channels', 'num_res_blocks', 'learn_sigma',
                    'he_feature_dim', 'st_feature_dim', 'diffusion_steps', 'noise_schedule'):
            if getattr(args, key) != trained[key]:
                raise ValueError(f'{model_dir}: {key} differs between training and inference')
        if trained.get('activation_diagnostics_version') != 'spatial_v1':
            raise ValueError(f'{model_dir}: checkpoint predates spatial task diagnostics')
        if not isinstance(trained.get('st_resolution_by_sample'), dict):
            raise ValueError(f'{model_dir}: missing source ST resolution record')
        if not isinstance(trained.get('st_resolution_sources'), dict):
            raise ValueError(f'{model_dir}: missing source ST resolution provenance')
        if trained['gene_start'] != start_gene:
            raise ValueError(f'{model_dir}: gene group name and saved offset disagree')
        script_name = f"{args.dataset_use}/{dir_name}"
        results_dir = os.path.join(Output, script_name)
        os.makedirs(results_dir, exist_ok=True)
        logger.configure(dir=results_dir)

        logger.log(f"\n=== Test {start_gene}-{start_gene + args.gene_num} ===")
        model, diffusion = sr_create_model_and_diffusion(args)
        model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
        model.to(dist_util.dev()).eval()
        if getattr(args, "use_fp16", False):
            model.convert_to_fp16()

        test_dataset = load_data(
            data_root=args.data_root,
            dataset_use=args.dataset_use,
            SR_times=args.SR_times,
            status="Test",
            gene_num=args.gene_num,
            all_gene=args.all_gene,
            pre_model_dir=pre_model_dir,
            gene_start=start_gene, split_manifest=args.split_manifest, split=args.split,
            reference_cells=args.reference_cells, seed=args.seed,
        )
        data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_patches)

        csv_path = os.path.join(results_dir, "metrics.csv")
        with open(csv_path, "w") as f:
            f.write("SampleID,RMSE,SSIM,CC\n")

        for j, batch in enumerate(data_loader):
            (
                SR_ST,
                spot_ST,
                WSI_5120,
                WSI_320,
                gene_class,
                Gene_index_map,
                metadata_feature,
                scale_gt,
                co_expression,
                WSI_mask,
                sc,
                scgpt,
                pre_he,
                complexity,
                original_st_bin_um,
            ) = batch

            last_HQST = noise_like(SR_ST) * args.noise_sigma

            model_kwargs = {
                "low_res": spot_ST.to(dist_util.dev()),
                "WSI_5120": WSI_5120.to(dist_util.dev()),
                "WSI_320": WSI_320.to(dist_util.dev()),
                "gene_class": gene_class.to(dist_util.dev()),
                "Gene_index_map": Gene_index_map.to(dist_util.dev()),
                "metadata_feature": metadata_feature.to(dist_util.dev()),
                "scale_gt": scale_gt.to(dist_util.dev()),
                "co_expression": co_expression.to(dist_util.dev()),
                "WSI_mask": WSI_mask.to(dist_util.dev()),
                "sc": sc.to(dist_util.dev()),
                "scgpt": scgpt.to(dist_util.dev()),
                "pre_he": pre_he.to(dist_util.dev()),
                "complexity": complexity.to(dist_util.dev()),
                "original_st_bin_um": original_st_bin_um.to(dist_util.dev()),
                "last_HQST": last_HQST.to(dist_util.dev()),


            }

            hr = SR_ST.permute(0, 2, 3, 1).cpu().numpy()

            method = getattr(args, "sampling_method", "ddpm")
            if method not in ("ddpm", "ddim"):
                raise ValueError(f"Unsupported sampling_method: {method}")
            sample_fn = diffusion.ddim_sample_loop if method == "ddim" else diffusion.p_sample_loop

            with torch.no_grad():
                result = sample_fn(
                    model,
                    (SR_ST.shape[0], args.gene_num, 256, 256),
                    clip_denoised=getattr(args, "clip_denoised", True),
                    model_kwargs=model_kwargs,
                )

            sample = result[0] if isinstance(result, tuple) else result
            pred = sample.permute(0, 2, 3, 1).cpu().numpy()[0]
            if not np.isfinite(pred).all():
                raise RuntimeError("Non-finite generated expression values")
            np.save(os.path.join(results_dir, f"prediction_{j}.npy"), pred)
            cls_name, sample_id, layer, patch_id = test_dataset.selected_patches[j]
            sample_key = cls_name, sample_id
            with open(os.path.join(results_dir, f"prediction_{j}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "tissue": cls_name, "sample_id": sample_id,
                    "layer": layer, "patch_id": patch_id,
                    "genes": test_dataset.sub_gene_info[sample_key]["namesN"],
                    "original_st_bin_um": test_dataset.sample_resolution[sample_key],
                    "resolution_source": test_dataset.sample_resolution_source[sample_key],
                    "checkpoint": model_path,
                    "sampling_method": method,
                    "reverse_steps": diffusion.num_timesteps,
                }, f, indent=2)



            gt = hr[0]

            rmse, ssim_v, cc = compute_metrics(gt, pred)
            with open(csv_path, "a") as f:
                f.write(f"{j},{rmse:.4f},{ssim_v:.4f},{cc:.4f}\n")

            save_sample_images(gt, pred, j, os.path.join(results_dir, "samples"))

        logger.log("Finished\n")
        gc.collect()
        torch.cuda.empty_cache()

# ========== 主入口 ==========
if __name__ == "__main__":
    
    main(Output)
