"""Extract one registered, observed five-gene example from 10x mouse-pup tiles.

The source tree is made from the public Xenium v1.6.0 mouse-pup H&E and
transcripts. This only selects existing measurements; it makes no predictions,
cell masks, or co-expression matrices.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse

GENES = ('Ttn', 'Car3', 'Des', 'Cav1', 'Myoz2')
TILE = '3_10'
SOURCE_PAGE = 'https://www.10xgenomics.com/datasets/mouse-pup-preview-data-xenium-mouse-tissue-atlassing-panel-1-standard'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def extract(source, destination):
    source, destination = Path(source), Path(destination)
    with (source / 'gene_metadata.csv').open(newline='') as stream:
        panel = list(csv.DictReader(stream))
    indices = []
    for gene in GENES:
        matches = [int(row['channel']) for row in panel
                   if row['Gene_Name'] == gene and row['feature_type'] == 'gene']
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one source panel channel for {gene}, got {matches}')
        indices.append(matches[0])
    he_path = source / 'WSI' / TILE / '5120_to256.npy'
    hr_path = source / 'HR_ST' / TILE / 'HR_ST_256.npz'
    spot_path = source / 'spot_ST' / TILE / 'spot_ST.npz'
    image = np.load(he_path)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError('Expected the registered uint8 H&E 10 um/pixel tile')
    high = sparse.load_npz(hr_path)
    low = sparse.load_npz(spot_path)
    if high.shape != (65536, len(panel)) or low.shape != (676, len(panel)):
        raise ValueError('Unexpected raw ST shape or panel width')
    hr = high[:, indices].toarray().T.reshape(5, 256, 256).astype(np.int32)
    spot = low[:, indices].toarray().T.reshape(5, 26, 26).astype(np.int32)
    if np.any(hr < 0) or np.any(spot < 0) or not np.array_equal(
            hr.sum((1, 2)), spot.sum((1, 2))):
        raise ValueError('High- and low-resolution counts do not conserve each selected gene')
    destination.mkdir(parents=True, exist_ok=True)
    he_out = destination / '3_10_he.png'
    st_out = destination / '3_10_observed.npz'
    Image.fromarray(image).save(he_out)
    np.savez_compressed(st_out, hr_counts=hr, spot_counts=spot,
                        genes=np.asarray(GENES), source_channels=np.asarray(indices, dtype=np.int16))
    provenance = {
        'kind': 'observed registered H&E / Xenium ST, no model prediction',
        'source_page': SOURCE_PAGE,
        'source_dataset': 'Xenium_V1_mouse_pup; Xenium Onboard Analysis 1.6.0; CC BY 4.0',
        'sample': 'one mouse pup; one 5 um sagittal section; one spatial tile',
        'tile_id': TILE, 'x_um': 3840, 'y_um': 12800, 'field_um': [2560, 2560],
        'he_mpp': 10, 'hr_bin_um': 10, 'spot_bin_um': 100,
        'genes': list(GENES), 'source_channels': indices,
        'gene_count_sums': hr.sum((1, 2)).astype(int).tolist(),
        'source_sha256': {str(p.relative_to(source)): sha256(p)
                          for p in (he_path, hr_path, spot_path, source / 'gene_metadata.csv')},
        'output_sha256': {p.name: sha256(p) for p in (he_out, st_out)},
        'split': 'one-sample observation only; no train/test split or independent evaluation',
    }
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing registered 10x processed tree')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'input')
    args = parser.parse_args()
    result = extract(args.source, args.output)
    print(json.dumps({'genes': result['genes'], 'source_channels': result['source_channels'],
                      'gene_count_sums': result['gene_count_sums']}, indent=2))
