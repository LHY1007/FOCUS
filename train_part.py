"""Gene-group training schedule for FOCUS.

Trains independent five-gene groups and writes checkpoints for each. Called by `train.py`
with the parameters assembled from `config/config_train.yaml`.
"""

import argparse
import yaml
import argparse, time, random
from guided_diffusion import dist_util, logger
from guided_diffusion.load_dataset import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    add_dict_to_argparser
)
import os
from guided_diffusion.train_util import TrainLoop
import numpy as np


def train_fun(args, log_dir, n, pre_model_dir):
    import torch
    from pathlib import Path
    from guided_diffusion.finetune import configure_finetuning
    if args.all_gene <= 0 or args.gene_num <= 0 or args.all_gene % args.gene_num:
        raise ValueError('all_gene must be a positive multiple of gene_num; define a separate config for a partial group')
    seed = getattr(args, 'seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    resume_group = None
    if args.resume_checkpoint:
        import re
        match = re.match(r'^G(\d+)-(\d+)_', Path(args.resume_checkpoint).parent.name)
        if not match:
            raise ValueError('Resume checkpoint must live in its Gstart-end training directory')
        resume_group = (int(match.group(1)), int(match.group(2)))
        first = (n - 1) * args.all_gene
        if (resume_group[0] < first or resume_group[1] > first + args.all_gene or
                resume_group[1] - resume_group[0] != args.gene_num or
                (resume_group[0] - first) % args.gene_num):
            raise ValueError('Resume checkpoint gene range is outside the requested groups')
    for i in range(args.all_gene // args.gene_num):
        gene_start = (n - 1) * args.all_gene + i * args.gene_num
        gene_end = gene_start + args.gene_num
        if resume_group and gene_start < resume_group[0]:
            continue
        if resume_group and gene_start == resume_group[0] and gene_end != resume_group[1]:
            raise ValueError('Resume checkpoint gene range differs from the selected group')
        resume_checkpoint = args.resume_checkpoint if resume_group == (gene_start, gene_end) else ''
        stamp = time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000000:09d}'
        save_dir = Path(log_dir) / args.dataset_use / f'{args.SR_times}X' / f'G{gene_start}-{gene_end}_{stamp}'
        logger.configure(dir=str(save_dir))
        # Save the actual runtime configuration beside the checkpoint.
        with (save_dir / 'config.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump({**vars(args), 'gene_start': gene_start}, f)
        data = load_data(
            data_root=args.data_root, dataset_use=args.dataset_use, status='Train',
            SR_times=args.SR_times, gene_num=args.gene_num, all_gene=args.all_gene,
            pre_model_dir=pre_model_dir, gene_start=gene_start,
            split_manifest=args.split_manifest, split=args.split,
            reference_cells=args.reference_cells, seed=seed)
        model, diffusion = sr_create_model_and_diffusion(args)
        init = getattr(args, 'init_checkpoint', '')
        if init:
            model.load_state_dict(torch.load(init, map_location='cpu', weights_only=True), strict=True)
        if getattr(args, 'finetune', False):
            if not init and not resume_checkpoint:
                raise ValueError('Fine-tuning requires init_checkpoint or an adapted resume_checkpoint')
            count = configure_finetuning(model, getattr(args, 'lora_rank', 8))
            logger.log(f'LoRA adapted {count} matrices; U-Net encoders are frozen')
        model.to(dist_util.dev()).train()
        schedule = create_named_schedule_sampler(args.schedule_sampler, diffusion)
        TrainLoop(
            model=model, diffusion=diffusion, data=data, batch_size=args.batch_size,
            microbatch=args.microbatch, lr=args.lr, ema_rate=args.ema_rate,
            log_interval=args.log_interval, save_interval=args.save_interval,
            resume_checkpoint=resume_checkpoint, use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth, schedule_sampler=schedule,
            weight_decay=args.weight_decay, lr_anneal_steps=args.lr_anneal_steps,
            SR_times=args.SR_times, epoch=args.epoch, max_steps=getattr(args, 'max_steps', 0)).run_loop()
        if getattr(args, 'finetune', False):
            from guided_diffusion.finetune import merged_state_dict
            torch.save(merged_state_dict(model), save_dir / 'merged.pt')

def load_superres_data(data_root,dataset_use,status,SR_times,gene_num,all_gene,gene_order,gene_name_order,pre_model_dir):
    # Load the super-resolution data using the specified directories
    return load_data(data_root=data_root,dataset_use=dataset_use,status=status,SR_times=SR_times,gene_num=gene_num,all_gene=all_gene,gene_order=gene_order,gene_name_order=gene_name_order,pre_model_dir=pre_model_dir)
