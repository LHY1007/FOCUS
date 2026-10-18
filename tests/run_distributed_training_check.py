"""Two-rank CPU check for sharding, conditional DDP, saving and resume.

Run with ``OMP_NUM_THREADS=1 FOCUS_NUM_WORKERS=0 torchrun --standalone
--nproc_per_node=2 tests/run_distributed_training_check.py --output /tmp/focus-ddp``.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset

from guided_diffusion import logger
import guided_diffusion.load_dataset as load_dataset
from guided_diffusion.train_util import TrainLoop


class EightItems(Dataset):
    def __init__(self):
        self.visited = []

    def __len__(self):
        return 8

    def __getitem__(self, index):
        self.visited.append(index)
        value = torch.full((1, 1, 1), float(index + 1))
        return (value,) * 15


class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Conv2d(1, 1, 1)
        self.optional = nn.Conv2d(1, 1, 1)

    def forward(self, x, t, **conditions):
        out = self.shared(x)
        if int(x[0, 0, 0, 0].item()) % 2 == 0:
            out = out + self.optional(x)
        return out


class ToyDiffusion:
    num_timesteps = 10

    def training_losses(self, model, x, t, ratio, model_kwargs=None):
        output = model(x, t, **model_kwargs)
        return {'loss': ((output - x * 2) ** 2).mean((1, 2, 3))}


def collate(batch):
    return tuple(torch.stack([item[k] for item in batch]) for k in range(15))


def run(output):
    rank = int(os.environ['RANK'])
    dist.init_process_group('gloo', init_method='env://')
    load_dataset.collate_patches = collate
    os.environ['FOCUS_NUM_WORKERS'] = '0'
    torch.manual_seed(0)
    output.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(output))
    data = EightItems()
    model = ConditionalModel()
    first = TrainLoop(model=model, diffusion=ToyDiffusion(), data=data,
                      batch_size=2, microbatch=1, lr=1e-3, ema_rate=0.9999,
                      log_interval=1, save_interval=2, resume_checkpoint='',
                      SR_times=10, epoch=2, max_steps=2, seed=7)
    first.run_loop()
    first_samples = data.visited.copy()
    dist.barrier()
    checkpoint = output / 'model000002.pt'
    optimizer = output / 'opt000002.pt'
    if not checkpoint.is_file() or not optimizer.is_file():
        raise AssertionError('Rank zero did not save both checkpoint files')
    restored_data = EightItems()
    restored = ConditionalModel()
    second = TrainLoop(model=restored, diffusion=ToyDiffusion(), data=restored_data,
                       batch_size=2, microbatch=1, lr=1e-3, ema_rate=0.9999,
                       log_interval=1, save_interval=3,
                       resume_checkpoint=str(checkpoint), SR_times=10,
                       epoch=2, max_steps=3, seed=7)
    second.run_loop()
    dist.barrier()
    flat = torch.cat([p.detach().flatten() for p in restored.parameters()])
    gathered = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flat)
    if not all(torch.equal(gathered[0], item) for item in gathered[1:]):
        raise AssertionError('Rank parameters differ after synchronized training')
    (output / f'rank{rank}.json').write_text(json.dumps({
        'rank': rank, 'first_epoch_samples': first_samples,
        'resumed_samples': restored_data.visited,
        'checkpoint_step': second.step,
        'parameters_agree': True,
    }))
    dist.barrier()
    if rank == 0:
        reports = [json.loads((output / f'rank{i}.json').read_text()) for i in range(2)]
        if set(reports[0]['first_epoch_samples']) & set(reports[1]['first_epoch_samples']):
            raise AssertionError('Distributed sampler duplicated a training item')
        if set(reports[0]['first_epoch_samples'] + reports[1]['first_epoch_samples']) != set(range(8)):
            raise AssertionError('Distributed sampler did not cover the dataset')
        print(json.dumps({'distributed_training_check': 'passed', 'reports': reports}))
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
