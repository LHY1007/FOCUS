"""Project 10x Xenium's existing cell or nucleus polygons onto an H&E tile.

This reads official vendor boundaries. It performs no image segmentation
and does not write FOCUS's cell_mask.npy, whose generator is maintained
separately.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def project(boundaries, x_um, y_um, field_um=2560, mpp=0.5):
    side = int(round(field_um / mpp))
    if abs(side * mpp - field_um) > 1e-6:
        raise ValueError('field_um must be an integer number of pixels')
    parquet = pq.ParquetFile(boundaries)
    selected = []
    # Polygon vertices are in Xenium morphology coordinates (micrometres).
    # Include an edge margin so a boundary-crossing cell keeps its vertices.
    margin = 100
    for index in range(parquet.num_row_groups):
        table = parquet.read_row_group(index, columns=['cell_id', 'vertex_x', 'vertex_y'])
        xs = table['vertex_x'].to_numpy()
        ys = table['vertex_y'].to_numpy()
        inside = ((xs >= x_um - margin) & (xs < x_um + field_um + margin) &
                  (ys >= y_um - margin) & (ys < y_um + field_um + margin))
        if inside.any():
            selected.append(table.filter(__import__('pyarrow').array(inside)).to_pandas())
    if not selected:
        raise ValueError('No official cell polygons in this field')
    import pandas as pd
    vertices = pd.concat(selected, ignore_index=True)
    labels = np.zeros((side, side), dtype=np.int32)
    cells = 0
    cell_ids = []
    for cell_id, group in vertices.groupby('cell_id', sort=True):
        points = np.rint(np.column_stack(((group['vertex_x'].to_numpy() - x_um) / mpp,
                                          (group['vertex_y'].to_numpy() - y_um) / mpp))).astype(np.int32)
        if len(points) < 3:
            continue
        if (points[:, 0].max() < 0 or points[:, 1].max() < 0 or
                points[:, 0].min() >= side or points[:, 1].min() >= side):
            continue
        cells += 1
        cell_ids.append(str(cell_id))
        cv2.fillPoly(labels, [points], int(cells))
    if cells == 0 or not labels.any():
        raise ValueError('No official cells projected into the field')
    return labels, cell_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundaries', type=Path, required=True)
    parser.add_argument('--x-um', type=float, required=True)
    parser.add_argument('--y-um', type=float, required=True)
    parser.add_argument('--field-um', type=float, default=2560)
    parser.add_argument('--mpp', type=float, default=0.5)
    parser.add_argument('--boundary-type', choices=('cell', 'nucleus'), default='cell')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels, cell_ids = project(args.boundaries, args.x_um, args.y_um, args.field_um, args.mpp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, labels)
    args.output.with_suffix('.ids.txt').write_text('\n'.join(cell_ids) + '\n', encoding='utf-8')
    digest = hashlib.sha256(args.boundaries.read_bytes()).hexdigest()
    print(json.dumps({'output': str(args.output), 'shape': labels.shape,
                      'boundary_type': args.boundary_type,
                      'official_instances_intersecting_tile': len(cell_ids),
                      'source_sha256': digest}))


if __name__ == '__main__':
    main()
