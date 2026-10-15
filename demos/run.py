"""Run: python demos/run.py

Read ten input PNG/NPZ pairs; write one PNG preview and one float32 NPZ per pair.
The checkpoint is downloaded once to ../models/, outside this demo folder.
The README describes the checkpoint and conditioning inputs used in this demo.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import yaml
from guided_diffusion.script_util import sr_create_model_and_diffusion

MODEL_URL = 'https://github.com/LHY1007/FOCUS/releases/download/demo-1gene-v1/focus-demo-1gene.pt'
MODEL_SHA256 = '6313dc713743c6d5c0e3f833b37473ae4d6f0674d6d21eaa5d38d0feb385a0c4'
KEYS = ('low_res', 'WSI_320', 'gene_class', 'Gene_index_map',
        'metadata_feature', 'scale_gt', 'co_expression', 'WSI_mask', 'sc', 'scgpt', 'pre_he')


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(override):
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    path = ROOT/'models'/'focus-demo-1gene.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print('Downloading the one-gene demo checkpoint (about 70 MB)...', flush=True)
        part = path.with_suffix('.part')
        try:
            with urlopen(MODEL_URL, timeout=60) as source, part.open('wb') as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            if checksum(part) != MODEL_SHA256:
                raise RuntimeError('Checkpoint checksum mismatch')
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    if checksum(path) != MODEL_SHA256:
        raise RuntimeError(f'Checkpoint checksum mismatch: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Optional local one-gene checkpoint for offline use')
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('FOCUS currently requires an NVIDIA CUDA GPU.')
    inputs = sorted((DEMO/'input').glob('*.npz'))
    if not inputs:
        raise FileNotFoundError(f'No NPZ inputs in {DEMO / "input"}')
    output = DEMO/'output'
    output.mkdir(exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    with (ROOT/'config/config_test.yaml').open(encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config.update(gene_num=1, all_gene=1, batch_size=1, timestep_respacing='4')
    model, diffusion = sr_create_model_and_diffusion(SimpleNamespace(**config))
    state = torch.load(checkpoint_path(cli.checkpoint), map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    del state
    for number, path in enumerate(inputs, 1):
        with np.load(path, allow_pickle=False) as data:
            conditions = {key: torch.from_numpy(np.array(data[key], dtype=np.float32))
                          .unsqueeze(0).cuda() for key in KEYS}
        with Image.open(path.with_suffix('.png')) as he:
            rgb = np.array(he.convert('RGB'), dtype=np.float32)
        if rgb.shape != (256, 256, 3):
            raise ValueError(f'{path.stem}.png: expected a 256 x 256 RGB H&E image')
        conditions['WSI_5120'] = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).cuda()
        if tuple(conditions['low_res'].shape) != (1, 1, 26, 26):
            raise ValueError(f'{path.name}: expected one expression channel of size 26 x 26')
        if conditions['sc'].shape[-1] != 1:
            raise ValueError(f'{path.name}: reference must have one target-gene channel')
        if not all(torch.isfinite(value).all() for value in conditions.values()):
            raise ValueError(f'{path.name}: non-finite input values')
        conditions['last_HQST'] = torch.randn((1, 1, 256, 256), device='cuda')
        with torch.inference_mode():
            prediction, _ = diffusion.p_sample_loop(
                model, (1, 1, 256, 256), clip_denoised=True, model_kwargs=conditions)
        if prediction.shape != (1, 1, 256, 256) or not torch.isfinite(prediction).all():
            raise RuntimeError(f'{path.name}: invalid single-channel prediction')
        image = prediction[0, 0].cpu().numpy().astype(np.float32)
        np.savez_compressed(output/f'{path.stem}.npz', prediction=image,
                            display_min=np.float32(-1), display_max=np.float32(1))
        preview = np.rint((np.clip(image, -1, 1) + 1) * 127.5).astype(np.uint8)
        Image.fromarray(preview).save(output/f'{path.stem}.png')
        print(f'{number}/{len(inputs)}  {path.stem} -> output/{path.stem}.png + {path.stem}.npz', flush=True)


if __name__ == '__main__':
    main()
