"""Evaluate one registered five-gene prediction against observed ST counts.

The reported metrics describe the supplied pair and its stated evaluation level.
They do not by themselves establish an independent test set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity


def evaluate(observed_path, prediction_path):
    with np.load(observed_path, allow_pickle=False) as observed, np.load(
            prediction_path, allow_pickle=False) as generated:
        obs_names = observed['genes'].astype(str).tolist()
        pred_names = generated['genes'].astype(str).tolist()
        if obs_names != pred_names or len(obs_names) != 5 or len(set(obs_names)) != 5:
            raise ValueError('Both files must name the same five genes in the same order')
        if str(observed['source_tile']) != str(generated['source_tile']):
            raise ValueError('Observed and predicted arrays must identify the same registered tile')
        counts = np.asarray(observed['observed_hr_counts'], dtype=np.float64)
        scale = float(observed['scale_gt'])
        pred = np.asarray(generated['prediction'], dtype=np.float64)
        if (counts.shape != (5, 256, 256) or pred.shape != counts.shape or
                not np.isfinite(counts).all() or not np.isfinite(pred).all() or
                scale <= 0 or not np.isfinite(scale)):
            raise ValueError('Expected finite [5,256,256] arrays and a positive observed scale')
        truth = counts / scale
        rows = []
        for name, target, estimate, total in zip(obs_names, truth, pred, counts.sum((1, 2))):
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
        return {
            'source_tile': str(observed['source_tile']),
            'evaluation_level': str(generated['evaluation_level']),
            'observed_scale': scale,
            'predicted_values': 'normalized expression; not transcript counts',
            'mean_rmse_normalized': average('rmse_normalized'),
            'mean_ssim_normalized': average('ssim_normalized'),
            'mean_pcc_normalized': average('pcc_normalized'),
            'per_gene': rows,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.observed, args.prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
