"""Evaluate registered five-gene expression on a stated specimen and scale.

Pixel metrics use the observed count map divided by its recorded training
scale. The optional H&E alignment metrics use the registered 256-pixel image.
A count-conserving low-input interpolation is available as a local baseline.
Predicted UMI totals require calibrated count output and are not inferred from
normalized diffusion predictions.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel, uniform_filter
from skimage.metrics import structural_similarity


def _gradient(array):
    array = np.asarray(array, dtype=np.float64)
    return np.hypot(sobel(array, axis=0, mode='reflect'),
                    sobel(array, axis=1, mode='reflect'))


def _alignment(rgb, prediction, window):
    if window < 3 or window % 2 != 1:
        raise ValueError('LNCC window must be an odd integer of at least three pixels')
    gray = (rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722) / 255.0
    he_grad = _gradient(gray)
    st_grad = _gradient(prediction.mean(axis=0))
    mx, my = uniform_filter(he_grad, window), uniform_filter(st_grad, window)
    cross = uniform_filter(he_grad * st_grad, window) - mx * my
    vx = np.maximum(uniform_filter(he_grad ** 2, window) - mx ** 2, 0)
    vy = np.maximum(uniform_filter(st_grad ** 2, window) - my ** 2, 0)
    lncc = float(np.mean(cross / (np.sqrt(vx * vy) + 1e-8)))
    norm_he = he_grad / max(float(he_grad.max()), 1e-8)
    norm_st = st_grad / max(float(st_grad.max()), 1e-8)
    return {
        'gradient_lncc_he_prediction': lncc,
        'gradient_ssim_he_prediction': float(structural_similarity(norm_he, norm_st, data_range=1.0)),
        'gradient_ssim_normalization': 'each modality divided by its own maximum gradient',
        'alignment_window_px': window,
    }


def _input_baseline(low_counts):
    if low_counts.shape != (5, 26, 26):
        raise ValueError('Observed low-resolution counts must have shape [5,26,26]')
    yy, xx = np.indices((256, 256))
    sy, sx = yy // 10, xx // 10
    widths = np.minimum(10, 256 - np.arange(26) * 10)
    area = widths[:, None] * widths[None, :]
    counts = low_counts[:, sy, sx] / area[sy, sx]
    if not np.allclose(counts.sum((1, 2)), low_counts.sum((1, 2)), atol=1e-5):
        raise ValueError('Input interpolation did not conserve observed UMI totals')
    return counts


def _complexity_strata(truth, pred, complexity):
    if (complexity.shape != (16, 16) or not np.isfinite(complexity).all() or
            np.any((complexity < 0) | (complexity > 1))):
        raise ValueError('Native morphology complexity must be finite [16,16] within [0,1]')
    eligible = np.flatnonzero(complexity.ravel() > 0)
    if len(eligible) < 3:
        return []
    order = eligible[np.argsort(complexity.ravel()[eligible], kind='stable')]
    squared = ((truth - pred) ** 2).reshape(5, 16, 16, 16, 16)
    patch_mse = squared.transpose(1, 3, 0, 2, 4).mean((2, 3, 4)).ravel()
    rows = []
    for label, subset in zip(('low', 'middle', 'high'), np.array_split(order, 3)):
        values = complexity.ravel()[subset]
        rows.append({'stratum': label, 'nonempty_patches': len(subset),
                     'complexity_min': float(values.min()), 'complexity_max': float(values.max()),
                     'rmse_normalized': float(np.sqrt(patch_mse[subset].mean()))})
    return rows


def _score(names, counts, pred, scale, source_tile, evaluation_level, rgb, window, complexity):
    if (counts.shape != (5, 256, 256) or pred.shape != counts.shape or
            not np.isfinite(counts).all() or not np.isfinite(pred).all() or
            scale <= 0 or not np.isfinite(scale)):
        raise ValueError('Expected finite [5,256,256] arrays and a positive observed scale')
    truth = counts / scale
    rows = []
    for name, target, estimate, total in zip(names, truth, pred, counts.sum((1, 2))):
        corr = float(np.corrcoef(target.ravel(), estimate.ravel())[0, 1]) if (
            target.std() > 0 and estimate.std() > 0) else None
        rows.append({
            'gene': name,
            'observed_umi_total': int(total),
            'rmse_normalized': float(np.sqrt(np.mean((target - estimate) ** 2))),
            'ssim_normalized': float(structural_similarity(target, estimate, data_range=1.0)),
            'pcc_normalized': corr,
        })
    def average(key):
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None
    result = {
        'source_tile': source_tile,
        'evaluation_level': evaluation_level,
        'observed_scale': scale,
        'predicted_values': 'normalized expression; UMI counts unavailable without calibrated count output',
        'mean_rmse_normalized': average('rmse_normalized'),
        'mean_ssim_normalized': average('ssim_normalized'),
        'mean_pcc_normalized': average('pcc_normalized'),
        'per_gene': rows,
    }
    if rgb is not None:
        result.update(_alignment(rgb, pred, window))
    if complexity is not None:
        result['native_complexity_terciles'] = _complexity_strata(truth, pred, complexity)
    return result


def evaluate(observed_path, prediction_path, he_path=None, include_input_baseline=False, alignment_window=9):
    observed_path, prediction_path = Path(observed_path), Path(prediction_path)
    provenance_path = observed_path.parent / 'provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    checksums = provenance.get('file_sha256', {})
    def verify_hash(path):
        expected = checksums.get(path.name)
        if expected:
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError(f'{path}: checksum differs from the paired input provenance')
        return bool(expected)
    observed_verified = verify_hash(observed_path)
    with np.load(observed_path, allow_pickle=False) as observed, np.load(
            prediction_path, allow_pickle=False) as generated:
        names = observed['genes'].astype(str).tolist()
        pred_names = generated['genes'].astype(str).tolist()
        if names != pred_names or len(names) != 5 or len(set(names)) != 5:
            raise ValueError('Both files must name the same five genes in the same order')
        source_tile = str(observed['source_tile'])
        if source_tile != str(generated['source_tile']):
            raise ValueError('Observed and predicted arrays must identify the same registered tile')
        source_record = provenance.get('source_tile', {})
        if isinstance(source_record, dict) and source_record.get('id', source_tile) != source_tile:
            raise ValueError('Observed tile differs from its paired input provenance')
        counts = np.asarray(observed['observed_hr_counts'], dtype=np.float64)
        scale = float(observed['scale_gt'])
        pred = np.asarray(generated['prediction'], dtype=np.float64)
        complexity = (np.asarray(observed['complexity'], dtype=np.float64)
                      if 'complexity' in observed.files else None)
        rgb = None
        if he_path is None and observed_path.with_suffix('.png').is_file():
            he_path = observed_path.with_suffix('.png')
        if he_path is not None:
            he_path = Path(he_path)
            he_verified = verify_hash(he_path)
            with Image.open(he_path) as image:
                rgb = np.asarray(image.convert('RGB'), dtype=np.float64)
            if rgb.shape != (256, 256, 3):
                raise ValueError('Registered H&E must be 256x256 RGB for alignment metrics')
        result = _score(names, counts, pred, scale, source_tile,
                        str(generated['evaluation_level']), rgb, alignment_window, complexity)
        result['input_provenance_verified'] = bool(observed_verified and rgb is not None and he_verified)
        if include_input_baseline:
            baseline_counts = _input_baseline(
                np.asarray(observed['observed_low_counts'], dtype=np.float64))
            result['input_interpolation_baseline'] = _score(
                names, counts, baseline_counts / scale, scale, source_tile,
                'same specimen; observed low-resolution input, not a trained model',
                rgb, alignment_window, complexity)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--he', type=Path, help='Registered 256x256 H&E; auto-detected beside observed NPZ')
    parser.add_argument('--input-baseline', action='store_true',
                        help='Compare count-conserving interpolation of the observed low-resolution input')
    parser.add_argument('--compare', action='append', default=[], metavar='LABEL=NPZ',
                        help='Evaluate another registered prediction with identical gene order and tile')
    parser.add_argument('--alignment-window', type=int, default=9)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.observed, args.prediction, args.he,
                      args.input_baseline, args.alignment_window)
    if args.compare:
        result['comparisons'] = {}
        for specification in args.compare:
            label, separator, path = specification.partition('=')
            if not separator or not label or not path or label in result['comparisons']:
                parser.error('--compare requires a unique LABEL=NPZ value')
            result['comparisons'][label] = evaluate(
                args.observed, Path(path), args.he, False, args.alignment_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
