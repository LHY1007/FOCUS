"""Run one observed mouse-pup tile through a real-condition FOCUS checkpoint.

The bundled checkpoint was fitted on this same tile; generated expression is
an execution example, not held-out accuracy or a manuscript result.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

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

MODEL_SHA256 = '618ba53577dcdecd2c383f34dd869e1f5e5fce19b278c5509eeb70576f5616e7'
GENES = ('Ttn', 'Car3', 'Des', 'Cav1', 'Myoz2')
KEYS = ('low_res', 'WSI_320', 'gene_class', 'Gene_index_map',
        'metadata_feature', 'scale_gt', 'co_expression', 'WSI_mask',
        'sc', 'scgpt', 'pre_he', 'complexity', 'original_st_bin_um')


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
    path = DEMO/'focus-5gene.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    if checksum(path) != MODEL_SHA256:
        raise RuntimeError(f'Checkpoint checksum mismatch: {path}')
    return path


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Use another compatible local five-gene checkpoint')
    parser.add_argument('--steps', type=int, default=1000, help='Reverse steps; default matches the manuscript')
    parser.add_argument('--input-dir', type=Path, default=DEMO/'input')
    parser.add_argument('--output-dir', type=Path, default=DEMO/'output')
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('FOCUS currently requires an NVIDIA CUDA GPU.')
    inputs = sorted(cli.input_dir.glob('*.npz'))
    if not inputs:
        raise FileNotFoundError(f'No NPZ inputs in {cli.input_dir}')
    provenance = json.loads((cli.input_dir/'provenance.json').read_text(encoding='utf-8'))
    if provenance.get('kind') != 'one real-condition train-tile reconstruction; no independent held-out test':
        raise ValueError('Input provenance must identify the real observed sample and its evaluation level')
    output = cli.output_dir
    output.mkdir(exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    with (ROOT/'config/config_test.yaml').open(encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not 2 <= cli.steps <= 1000:
        parser.error('--steps must be between 2 and 1000')
    config.update(gene_num=5, all_gene=5, batch_size=1, manuscript_mode=True, learn_sigma=False,
                  predict_xstart=False, timestep_respacing=str(cli.steps))
    model, diffusion = sr_create_model_and_diffusion(SimpleNamespace(**config))
    state = torch.load(checkpoint_path(cli.checkpoint), map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    del state
    torch.cuda.reset_peak_memory_stats()
    for number, path in enumerate(inputs, 1):
        if provenance['file_sha256'].get(path.name) != checksum(path):
            raise ValueError(f'{path.name}: input hash disagrees with provenance')
        he_path = path.with_suffix('.png')
        if provenance['file_sha256'].get(he_path.name) != checksum(he_path):
            raise ValueError(f'{he_path.name}: H&E hash disagrees with provenance')
        with np.load(path, allow_pickle=False) as data:
            genes = np.array(data['genes'])
            if genes.tolist() != list(GENES):
                raise ValueError(f'{path.name}: expected checkpoint gene order {GENES}')
            observed_hr = np.array(data['observed_hr_counts'])
            observed_low = np.array(data['observed_low_counts'])
            if (observed_hr.shape != (5, 256, 256) or observed_low.shape != (5, 26, 26) or
                    not np.array_equal(observed_hr.sum((1, 2)), observed_low.sum((1, 2)))):
                raise ValueError(f'{path.name}: observed H&E/ST pairing failed count conservation')
            conditions = {key: torch.from_numpy(np.array(data[key], dtype=np.float32))
                          .unsqueeze(0).cuda() for key in KEYS}
        source_resolution = provenance.get('original_st_resolution', {})
        if (float(conditions['original_st_bin_um'].item()) !=
                source_resolution.get('original_st_bin_um') or
                not str(source_resolution.get('source', '')).strip()):
            raise ValueError(f'{path.name}: physical ST resolution disagrees with source record')
        with Image.open(he_path) as he:
            rgb = np.array(he.convert('RGB'), dtype=np.float32)
        if rgb.shape != (256, 256, 3):
            raise ValueError(f'{path.stem}.png: expected a 256 x 256 RGB H&E image')
        conditions['WSI_5120'] = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).cuda()
        if tuple(conditions['low_res'].shape) != (1, 5, 26, 26):
            raise ValueError(f'{path.name}: expected five expression channels of size 26 x 26')
        if conditions['sc'].shape[-1] != 5:
            raise ValueError(f'{path.name}: reference must have five target-gene channels')
        expected = {'WSI_320': (1, 256, 3, 16, 16), 'metadata_feature': (1, 768),
                    'Gene_index_map': (1, 5, 256, 256), 'scgpt': (1, 512, 26, 26),
                    'pre_he': (1, 257, 1536), 'complexity': (1, 16, 16),
                    'original_st_bin_um': (1,)}
        for key, shape in expected.items():
            if tuple(conditions[key].shape) != shape:
                raise ValueError(f'{path.name}: {key} must have shape {shape}')
        if not all(torch.isfinite(value).all() for value in conditions.values()):
            raise ValueError(f'{path.name}: non-finite input values')
        if any(not torch.count_nonzero(conditions[key]) for key in
               ('WSI_mask', 'sc', 'scgpt', 'pre_he', 'complexity', 'original_st_bin_um')):
            raise ValueError(f'{path.name}: missing or placeholder multimodal conditions')
        conditions['last_HQST'] = torch.zeros((1, 5, 256, 256), device='cuda')
        with torch.inference_mode():
            prediction = diffusion.ddim_sample_loop(
                model, (1, 5, 256, 256), clip_denoised=True, model_kwargs=conditions)
        if prediction.shape != (1, 5, 256, 256) or not torch.isfinite(prediction).all():
            raise RuntimeError(f'{path.name}: invalid five-channel prediction')
        image = prediction[0].clamp(0, 1).cpu().numpy().astype(np.float32)
        np.savez_compressed(output/f'{path.stem}.npz', prediction=image, genes=genes,
                            source_tile=np.asarray(path.stem), steps=np.int32(cli.steps),
                            evaluation_level=np.asarray('training-tile reconstruction'),
                            display_min=np.float32(0), display_max=np.float32(1))
        preview = np.concatenate([np.rint(plane * 255).astype(np.uint8) for plane in image], axis=1)
        Image.fromarray(preview).save(output/f'{path.stem}.png')
        print(f'{number}/{len(inputs)}  {path.stem} -> output/{path.stem}.png + {path.stem}.npz', flush=True)
    print(f'CUDA peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; '
          f'peak reserved: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB', flush=True)
    print(f'Elapsed: {time.perf_counter() - started:.2f} s', flush=True)


if __name__ == '__main__':
    main()
