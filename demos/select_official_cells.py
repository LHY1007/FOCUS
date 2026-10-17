"""Pack a small, spatially spread set of existing Xenium cell IDs for the demo.

Reads 10x's native cell-instance labels. Does not segment H&E, infer cells, or
change the repository's cell-mask generation pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects


def select(label_path, complexity_path, output):
    label_path, complexity_path, output = map(Path, (label_path, complexity_path, output))
    if output.exists():
        raise FileExistsError(output)
    labels = np.load(label_path, mmap_mode='r')
    ids = label_path.with_suffix('.ids.txt').read_text(encoding='utf-8').splitlines()
    complexity = np.load(complexity_path)
    if (labels.shape != (5120, 5120) or complexity.shape != (16, 16) or
            len(ids) != int(labels.max())):
        raise ValueError('Expected one registered native 2,560 um field and official cell ID map')
    areas = np.bincount(labels.ravel(), minlength=len(ids) + 1)
    sx = np.zeros(len(areas), dtype=np.float64)
    sy = np.zeros(len(areas), dtype=np.float64)
    xs = np.arange(5120, dtype=np.float64)
    for y0 in range(0, 5120, 128):
        block = np.asarray(labels[y0:y0 + 128])
        sx += np.bincount(block.ravel(), weights=np.broadcast_to(xs, block.shape).ravel(),
                          minlength=len(areas))
        sy += np.bincount(block.ravel(), weights=np.broadcast_to(
            np.arange(y0, y0 + 128, dtype=np.float64)[:, None], block.shape).ravel(),
            minlength=len(areas))
    cy = sy / np.maximum(areas, 1)
    cx = sx / np.maximum(areas, 1)
    chosen = []
    for row in range(8):
        for col in range(8):
            candidates = np.flatnonzero(
                (areas >= 20) & (cy >= row * 640) & (cy < (row + 1) * 640) &
                (cx >= col * 640) & (cx < (col + 1) * 640))
            candidates = candidates[candidates != 0]
            if not len(candidates):
                raise ValueError(f'No existing cell labels in grid {row},{col}')
            chosen.append(int(candidates[np.argmax(areas[candidates])]))
    # The consistency loss uses the most complex 160 um field. Ensure it has
    # several real cells after downsampling, independent of the 8x8 selection.
    row, col = np.unravel_index(complexity.argmax(), complexity.shape)
    candidates = np.flatnonzero((areas >= 20) & (cy >= row * 320) & (cy < (row + 1) * 320) &
                                (cx >= col * 320) & (cx < (col + 1) * 320))
    candidates = candidates[candidates != 0]
    extras = sorted(candidates, key=lambda i: (-areas[i], i))[:4]
    if len(extras) < 4:
        raise ValueError('Too few official cells in the most complex field')
    chosen = list(dict.fromkeys(chosen + [int(i) for i in extras]))
    slices = find_objects(np.asarray(labels))
    masks = np.zeros((256, 256, len(chosen)), dtype=np.uint8)
    for position, label in enumerate(chosen):
        box = slices[label - 1]
        yy, xx = np.where(np.asarray(labels[box]) == label)
        ybins = (yy + box[0].start) // 20
        xbins = (xx + box[1].start) // 20
        masks[ybins, xbins, position] = 1
    if np.any(masks.sum((0, 1)) == 0):
        raise ValueError('Selected cell vanished at 10 um/pixel')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, masks)
    summary = {
        'kind': 'read-only selection and 10 um occupancy projection of official 10x cell polygons',
        'visible_official_cells': int((areas[1:] > 0).sum()),
        'selected_cells': len(chosen),
        'official_cell_ids': [ids[i - 1] for i in chosen],
        'native_label_values': chosen,
        'maximum_complexity_field': [int(row), int(col)],
        'cells_in_maximum_field': int(sum(
            masks[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16, j].any()
            for j in range(len(chosen)))),
        'source_label_sha256': hashlib.sha256(label_path.read_bytes()).hexdigest(),
    }
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--complexity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = select(args.labels, args.complexity, args.output)
    print(json.dumps({k: result[k] for k in
                      ('visible_official_cells', 'selected_cells', 'maximum_complexity_field',
                       'cells_in_maximum_field')}, indent=2))
