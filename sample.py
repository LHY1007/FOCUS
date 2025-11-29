import os, re, glob, gc, yaml, torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mpi4py import MPI
from guided_diffusion import dist_util, logger
from guided_diffusion.load_dataset import load_data
from guided_diffusion.script_util import sr_create_model_and_diffusion, add_dict_to_argparser
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
gpu_ids = [0]
torch.cuda.set_device(gpu_ids[rank])
def create_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    with open("./code/config/config_test.yaml", "r") as file:
        config = yaml.safe_load(file)
    add_dict_to_argparser(parser, config)
    return parser

args = create_argparser().parse_args()
args.all_gene = 5
args.gene_num = 5
args.batch_size = 1
args.SR_times = 10
args.dataset_use = "All"
args.data_root = "/home/cbtil4/ST/datasets/demo/"
args.noise_sigma = 1.0
Output = "TEST_Result-demo"
global_num = "_demo"

# ========== 辅助函数 ==========
def inv_l1_log(sr_norm_patch, patch_scale):
    sr_lognorm = sr_norm_patch * patch_scale[..., None, None, None]
    sr_counts_norm = (np.expm1(sr_lognorm)) / 1e4
    return sr_counts_norm

def noise_like(tensor):
    return torch.randn_like(tensor)

def normalize_prediction(pred):
    for k in range(pred.shape[-1]):
        d = pred[..., k]
        if np.max(d) > np.min(d):
            pred[..., k] = (d - np.min(d)) / (np.max(d) - np.min(d))
    return pred

def compute_metrics(gt, pred):
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    try:
        ssim_v = ssim(gt[..., 0], pred[..., 0], data_range=1.0)
    except:
        ssim_v = 0
    cc = np.corrcoef(gt.flatten(), pred.flatten())[0, 1]
    return rmse, ssim_v, cc

def save_sample_images(gt, pred, idx, outdir):
    os.makedirs(outdir, exist_ok=True)
    for i in range(gt.shape[-1]):
        plt.imsave(os.path.join(outdir, f"{idx}_{i}gt.png"), gt[..., i], cmap="viridis")
        plt.imsave(os.path.join(outdir, f"{idx}_{i}pred.png"), pred[..., i], cmap="viridis")


def main(Output):
    pre_model_dir  = '/model'
    model_dirs = sorted(glob.glob(os.path.join(f"./logs{global_num}", args.dataset_use,"10X", "G*")))

    if len(model_dirs) == 0:
        print("⚠️ unable to find model directories.")
        return

    for model_dir in model_dirs:
        dir_name = os.path.basename(model_dir)
        g_part = dir_name.split("G")[1].split("_")[0]
        start_gene = int(g_part.split("-")[0])

        ckpts = glob.glob(os.path.join(model_dir, "model*.pt"))
        if not ckpts:
            continue
        max_step = max(int(re.search(r"model(\d+)\.pt", ck).group(1)) for ck in ckpts)
        model_path = os.path.join(model_dir, f"model{max_step:06d}.pt")

        script_name = f"{args.dataset_use}/G{start_gene}-{start_gene + args.gene_num}"
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
            pre_model_dir=pre_model_dir
        )
        data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

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
                "last_HQST": last_HQST.to(dist_util.dev()),


            }

            hr = SR_ST.permute(0, 2, 3, 1).cpu().numpy()

            sample_fn = (
                diffusion.ddim_sample_loop
                if getattr(args, "sampling_method", "ddpm") == "ddim"
                else diffusion.p_sample_loop
            )

            with torch.no_grad():
                sample, scale_pred = sample_fn(
                    model,
                    (args.batch_size, args.gene_num, 256, 256),
                    clip_denoised=getattr(args, "clip_denoised", True),
                    model_kwargs=model_kwargs,
                )

            pred = sample.permute(0, 2, 3, 1).cpu().numpy()[0]



            pred = normalize_prediction(pred)
            gt = hr[0]

            rmse, ssim_v, cc = compute_metrics(gt, pred)
            with open(csv_path, "a") as f:
                f.write(f"{j},{rmse:.4f},{ssim_v:.4f},{cc:.4f}\n")

            save_sample_images(gt, pred, j, os.path.join(results_dir, "samples"))

        logger.log("✅ Finished\n")
        gc.collect()
        torch.cuda.empty_cache()

# ========== 主入口 ==========
if __name__ == "__main__":
    
    main(Output)
