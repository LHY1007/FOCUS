"""Patient-level split assignment and validation before patch extraction.

Input CSV: tissue,sample_id,patient_id,split. The first three columns must be
real source identifiers. An empty split can be assigned by the CLI.
"""
import argparse
import csv
from pathlib import Path
import numpy as np

SPLITS = {'train', 'validation', 'finetune', 'test'}


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError('Empty sample manifest')
    patients, samples = {}, {}
    for row in rows:
        for key in ('tissue', 'sample_id', 'patient_id', 'split'):
            if not row.get(key, '').strip():
                raise ValueError(f'Missing {key}: {row}')
        if row['split'] not in SPLITS:
            raise ValueError(f'Unknown split: {row["split"]}')
        key = (row['tissue'], row['sample_id'])
        if key in samples:
            raise ValueError(f'Duplicate sample: {key}')
        if row['patient_id'] in patients and patients[row['patient_id']] != row['split']:
            raise ValueError(f'Patient leakage across splits: {row["patient_id"]}')
        patients[row['patient_id']] = row['split']
        samples[key] = row
    return samples


def assign_patients(rows, seed=0, fractions=(0.8, 0.1, 0.1)):
    if len(fractions) != 3 or any(x < 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError('train/validation/test fractions must be nonnegative and sum to 1')
    for row in rows:
        if any(not row.get(k, '').strip() for k in ('tissue', 'sample_id', 'patient_id')):
            raise ValueError('Supply real tissue, sample_id and patient_id for every sample')
        if row.get('split', '').strip():
            raise ValueError('Assignment requires empty splits; validate existing splits without --assign')
    patients = sorted({r['patient_id'] for r in rows})
    np.random.default_rng(seed).shuffle(patients)
    counts = np.array(fractions) * len(patients)
    sizes = np.floor(counts).astype(int)
    for i in np.argsort(-(counts - sizes))[:len(patients) - sizes.sum()]:
        sizes[i] += 1
    assignments, start = {}, 0
    for split, count in zip(('train', 'validation', 'test'), sizes):
        assignments.update({p: split for p in patients[start:start + count]})
        start += count
    return [{**r, 'split': assignments[r['patient_id']]} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--assign', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fractions', nargs=3, type=float, default=(0.8, 0.1, 0.1))
    args = parser.parse_args()
    if args.assign:
        if not args.output or Path(args.output).exists():
            parser.error('--assign requires a new --output path')
        with Path(args.manifest).open(encoding='utf-8-sig', newline='') as f:
            rows = assign_patients(list(csv.DictReader(f)), args.seed, args.fractions)
        with Path(args.output).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=('tissue', 'sample_id', 'patient_id', 'split'))
            writer.writeheader()
            writer.writerows(rows)
        args.manifest = args.output
    samples = read_manifest(args.manifest)
    print(f'Validated {len(samples)} samples from {len({r["patient_id"] for r in samples.values()})} patients; no patient crosses splits.')


if __name__ == '__main__':
    main()
