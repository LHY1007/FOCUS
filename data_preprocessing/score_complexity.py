"""Score an existing native-resolution cell segmentation; does not make masks.

Example: python data_preprocessing/score_complexity.py --he native_he.tif
    --labels native_cell_labels.npy --output complexity_grid.npy
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided_diffusion.morphology import complexity_from_native_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--he', type=Path, required=True, help='Registered native RGB H&E image')
    parser.add_argument('--labels', type=Path, required=True, help='Existing integer cell-instance labels')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pixel-size-um', type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    image = np.load(args.he) if args.he.suffix == '.npy' else np.asarray(Image.open(args.he).convert('RGB'))
    labels = np.load(args.labels)
    if abs(image.shape[0] * args.pixel_size_um - 2560) > 1e-4:
        raise ValueError('The native H&E field must cover 2,560 um at the supplied pixel size')
    score = complexity_from_native_labels(image, labels, args.pixel_size_um)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, score)
    print(f'Saved {args.output}: 16x16 density and morphology complexity from {len(np.unique(labels))-1} labels')


if __name__ == '__main__':
    main()
