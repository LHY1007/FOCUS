"""Step 7 -- stitch the low-resolution tiles back into a whole-slide preview.

``step5_extract_he_tiles.py`` writes each tile twice: the full 5120 px crop and a
256 px thumbnail. The thumbnails are laid out on the same 50 % overlapping grid,
so putting them back in place reconstructs a downsampled version of the slide.
That preview is the quickest way to confirm that the tiling grid, the tile
ordering and the registration all agree with the tissue.

Tiles are 50 % overlapping, so the last writer wins in the shared border. That is
fine for a preview; the dataloader treats tiles as independent samples and never
needs the stitch.

    python step7_stitch_preview.py
    python step7_stitch_preview.py --no-transform
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from config import sample

#: Folder names are ``{i}_{j}`` with the tile indices of the covering grid.
TILE_PATTERN = re.compile(r"^(\d+)_(\d+)$")


def list_tiles(wsi_root: Path) -> list[tuple[int, int]]:
    """Return the ``(i, j)`` indices of every tile folder under *wsi_root*.

    Folders that do not match ``{i}_{j}`` are ignored, which keeps auxiliary
    entries such as ``gene_names.csv`` out of the stitch.
    """
    positions = []
    for entry in sorted(wsi_root.iterdir()):
        if not entry.is_dir():
            continue
        match = TILE_PATTERN.match(entry.name)
        if match:
            positions.append((int(match.group(1)), int(match.group(2))))
    return positions


def stitch(wsi_root: Path, tile_name: str, transform: bool) -> np.ndarray:
    """Reassemble the slide from the ``tile_name`` thumbnails under *wsi_root*.

    *transform* applies the same quarter-turn and horizontal flip the slide was
    viewed with when the tiles were cut, so the preview comes out upright.
    """
    positions = list_tiles(wsi_root)
    if not positions:
        raise FileNotFoundError(f"no {{i}}_{{j}} tile folders under {wsi_root}")

    max_i = max(i for i, _ in positions)
    max_j = max(j for _, j in positions)

    step = config.SMALL_SIDE * 2          # thumbnails overlap by half a tile
    half = config.SMALL_SIDE
    full_width = (max_i + 1) * step + half
    full_height = (max_j + 1) * step + half

    canvas = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    for i, j in tqdm(positions, desc="stitching"):
        path = wsi_root / f"{i}_{j}" / tile_name
        if not path.exists():
            continue
        tile = np.load(path)
        if tile.shape[0] != config.SMALL_SIDE or tile.shape[1] != config.SMALL_SIDE:
            print(f"  skip {i}_{j}: unexpected shape {tile.shape}")
            continue

        x = i * half
        y = j * half
        x_end = min(x + config.SMALL_SIDE, full_width)
        y_end = min(y + config.SMALL_SIDE, full_height)
        canvas[y:y_end, x:x_end] = tile[:y_end - y, :x_end - x]

    if transform:
        canvas = np.rot90(canvas, k=3)
        canvas = np.flip(canvas, axis=1)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=None,
                        help="key in config.SAMPLES (default: ACTIVE_SAMPLE)")
    parser.add_argument("--tile", default="5120_to256.npy",
                        help="which per-tile array to stitch")
    parser.add_argument("--out", default="restored_WSI_256.png",
                        help="file name written inside the WSI root")
    parser.add_argument("--no-transform", action="store_true",
                        help="skip the quarter-turn and flip")
    args = parser.parse_args()

    cfg = sample(args.sample)
    wsi_root = Path(cfg["wsi_root"])

    canvas = stitch(wsi_root, args.tile, transform=not args.no_transform)
    out_path = wsi_root / args.out
    cv2.imwrite(str(out_path), canvas)
    print(f"wrote {out_path}  {canvas.shape[1]} x {canvas.shape[0]}")


if __name__ == "__main__":
    main()
