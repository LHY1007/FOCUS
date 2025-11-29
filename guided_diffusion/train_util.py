import copy
import functools
import os
import sys
import blobfile as bf
import numpy as np
import torch
import torch as th
import torch.distributed as dist
from scipy.stats import truncnorm
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from collections import Counter
import time
from torch.utils.data import DataLoader
from guided_diffusion import dist_util, logger
#from . import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
#from .fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
#from .nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
#from .resample import LossAwareSampler, UniformSampler
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def np2tensor(arr):
    return th.from_numpy(np.array(arr)).float()

import time
import functools
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


class TrainLoop:

    
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        SR_times=10,
        epoch=1000
    ):
        # ==================== 基本参数 ====================
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.SR_times = SR_times
        self.epoch = epoch
        
        # ==================== 训练状态 ====================
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        
        # ==================== EMA 配置 ====================
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        
        # ==================== 日志和保存配置 ====================
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        
        # ==================== 混合精度训练 ====================
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        # ==================== 采样器 ====================
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        
        # ==================== 设备配置 ====================
        self.sync_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.sync_cuda else "cpu")
        
        # ==================== 加载模型和优化器 ====================
        self._load_parameters()
        
        print(f'Number of Trainable Parameters: {count_parameters(model)}')
        
        self.opt = AdamW(
            self.mp_trainer.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 不使用 DDP 包装
        self.ddp_model = self.model
        
        # ==================== 数据加载器 ====================
        self.data_loader = self._create_dataloader_with_prefetch()
    
    def _load_parameters(self):
        """加载模型检查点"""
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Loading model from checkpoint: {resume_checkpoint}...")
            
            state_dict = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict)
    
    def _load_optimizer_state(self):
        """加载优化器状态"""
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint),
            f"opt{self.resume_step:06}.pt"
        )
        
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(opt_checkpoint, map_location=self.device)
            self.opt.load_state_dict(state_dict)
    
    def _create_base_loader(self, deterministic=False):
        """
        创建基础数据加载器
        
        Args:
            deterministic: 是否使用确定性加载 (不shuffle)
        """
        num_workers = min(32, os.cpu_count() // 4)
        
        loader = DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=not deterministic,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True,
        )
        
        # 无限循环迭代器
        while True:
            yield from loader
    
    def _create_dataloader_with_prefetch(self, max_prefetch=4):
        """
        创建带预取功能的数据加载器
        
        Args:
            max_prefetch: 预取队列的最大长度
        """
        base_loader = self._create_base_loader()
        prefetch_queue = []
        
        # 初始化预取队列
        for _ in range(max_prefetch):
            try:
                batch_data = self._fetch_and_prepare_batch(base_loader)
                if batch_data:
                    prefetch_queue.append(batch_data)
            except StopIteration:
                break
        
        # 生成批次并维护预取队列
        while prefetch_queue:
            # 返回队列头部的批次
            yield prefetch_queue.pop(0)
            
            # 向队列尾部添加新批次
            try:
                batch_data = self._fetch_and_prepare_batch(base_loader)
                if batch_data:
                    prefetch_queue.append(batch_data)
            except StopIteration:
                continue
    
    def _fetch_and_prepare_batch(self, loader):
        """
        从加载器获取批次并准备模型输入
        
        Returns:
            tuple: (SR_ST, model_kwargs) 或 None
        """
        try:
            SR_ST, spot_ST, WSI_5120, WSI_320, gene_class, Gene_index_map, metadata_feature, scale_gt, co_expression, WSI_mask, sc, scgpt, pre_he = next(loader)

   
            model_kwargs = {
                "low_res": spot_ST,
                "WSI_5120": WSI_5120,
                "WSI_320": WSI_320,
                "gene_class": gene_class,
                "Gene_index_map": Gene_index_map,
                "metadata_feature": metadata_feature,
                "scale_gt": scale_gt,
                "co_expression": co_expression,
                "WSI_mask": WSI_mask,
                "sc": sc,
                "scgpt": scgpt,
                "pre_he": pre_he
            }
            
            return (SR_ST, model_kwargs)
        except StopIteration:
            return None
    
    def _calculate_training_ratio(self, current_step, total_steps):
        """
        根据训练进度计算数据使用比例
        
        Args:
            current_step: 当前步数
            total_steps: 总步数
        
        Returns:
            float: 数据使用比例 (0.1 到 1.0)
        """
        ratio_schedule = [
            (0.1, 0.1),
            (0.2, 0.1),
            (0.3, 0.3),
            (0.4, 0.5),
            (0.5, 0.7),
            (0.6, 0.9),
            (0.7, 1.0),
            (0.8, 1.0),
            (0.9, 1.0),
            (1.0, 1.0),
        ]
        
        progress = current_step / total_steps
        
        for threshold, ratio in ratio_schedule:
            if progress < threshold:
                return ratio
        
        return 1.0
    
    def _format_time(self, seconds):
        """
        格式化时间为 HH:MM:SS
        
        Args:
            seconds: 秒数
        
        Returns:
            str: 格式化的时间字符串
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    
    def run_loop(self):
        """主训练循环"""
        # 计算总迭代次数
        total_iterations = int(99 * self.epoch / self.batch_size) + 1
        
        # 初始化计时
        loop_start_time = time.time()
        
        logger.log(f"Starting training loop: {total_iterations} iterations")
        
        while self.step <= total_iterations:
            step_start_time = time.time()
            
            # 计算当前训练比例
            ratio = self._calculate_training_ratio(self.step, total_iterations)
            
            # 获取批次并执行训练步骤
            batch, cond = next(self.data_loader)
            self.run_step(batch, cond, ratio)
            
            # 计算时间统计
            current_time = time.time()
            step_duration = current_time - step_start_time
            total_duration = current_time - loop_start_time
            
            # 估算剩余时间
            remaining_steps = total_iterations - self.step
            avg_time_per_step = total_duration / (self.step + 1)
            estimated_remaining = avg_time_per_step * remaining_steps
            
            # 日志输出
            if self.step % self.log_interval == 0:
                logger.log(
                    f"Step {self.step}/{total_iterations} | "
                    f"Ratio: {ratio:.1f} | "
                    f"Elapsed: {self._format_time(total_duration)} | "
                    f"Remain: {self._format_time(estimated_remaining)} | "
                    f"Speed: {step_duration:.2f}s/step"
                )
                logger.dumpkvs()
            
            # 保存检查点
            if self.step % self.save_interval == 0 and self.step != 0:
                self.save()
                
                # 测试模式下提前退出
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            
            self.step += 1
        
        # 保存最终检查点
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        
        logger.log("Training completed!")
    
    def run_step(self, batch, cond, ratio):
        """
        执行单个训练步骤
        
        Args:
            batch: 输入批次
            cond: 条件信息
            ratio: 训练数据使用比例
        """
        # 前向传播和反向传播
        self.forward_backward(batch, cond, ratio)
        
        # 优化器步骤
        took_step = self.mp_trainer.optimize(self.opt)
        
        # 学习率退火
        self._anneal_lr()
        
        # 记录日志
        self.log_step()
    
    def forward_backward(self, batch, cond, ratio):
        """
        执行前向传播和反向传播
        
        Args:
            batch: 输入批次 [B, C, H, W]
            cond: 条件字典
            ratio: 训练数据使用比例
        """
        self.mp_trainer.zero_grad()
        
        # 微批次处理
        for i in range(0, batch.shape[0], self.microbatch):
            # 提取微批次
            micro = batch[i:i + self.microbatch].to(self.device)
            
            # SR_times=5 时需要插值到 256x256
            if self.SR_times == 5:
                micro = F.interpolate(micro, size=(256, 256))
            
            # 准备微批次条件
            micro_cond = {
                k: v[i:i + self.microbatch].to(self.device)
                for k, v in cond.items()
            }
            
            # 判断是否为最后一个微批次
            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            # 采样时间步
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            
            # 计算损失
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                ratio,
                model_kwargs=micro_cond,
            )
            
            losses = compute_losses()
            
            # 更新采样器
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            # 加权平均损失
            loss = (losses["loss"] * weights).mean()
            
            # 记录损失
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            
            # 反向传播
            self.mp_trainer.backward(loss)
    
    def _anneal_lr(self):
        """学习率退火"""
        if not self.lr_anneal_steps:
            return
        
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
    
    def log_step(self):
        """记录当前步骤的统计信息"""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
    
    def save(self):
        """保存模型检查点"""
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
                logger.log(f"Saving model checkpoint: {filename}")
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)
        
        save_checkpoint(0, self.mp_trainer.master_params)
    
    def save_optimizer_state(self):
        """保存优化器状态 (可选功能)"""
        filename = f"opt{(self.step + self.resume_step):06d}.pt"
        logger.log(f"Saving optimizer state: {filename}")
        
        with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
            torch.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
