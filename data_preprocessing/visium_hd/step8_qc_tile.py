"""Step 8 -- per-tile quality control.

For one tile this prints the shapes and sparsity of the three transcript grids
and renders the H&E thumbnail next to the mean-expression map of the 10 um grid.
The two images must line up: a bright band in the expression map that sits on a
dark fold in the H&E (or the reverse) means the registration matrix is wrong for
that slide.

A figure is written next to the tile unless ``--no-figure`` is given.

    python step8_qc_tile.py --tile 3_3
    python step8_qc_tile.py --tile 3_3 --out qc/3_3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.sparse import load_npz

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  - backend selected above

import config
from config import sample


def normalise(image: np.ndarray) -> np.ndarray:
    """Scale *image* to ``[0, 1]``; a constant image is returned as zeros."""
    low, high = float(np.min(image)), float(np.max(image))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return (image - low) / (high - low)


def describe(name: str, path: Path, grid_side: int) -> np.ndarray | None:
    """Print a short summary of one ``.npz`` grid and return its dense form."""
    if not path.exists():
        print(f"  {name}: missing ({path})")
        return None

    matrix = load_npz(path)
    expected = grid_side * grid_side
    ok = "ok" if matrix.shape[0] == expected else f"EXPECTED {expected} ROWS"
    total = float(matrix.sum())
    print(f"  {name}: {matrix.shape[0]} x {matrix.shape[1]}  "
          f"grid {grid_side}x{grid_side} [{ok}]  "
          f"nnz {matrix.nnz}  total {total:.0f}")
    return matrix.toarray()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=None,
                        help="key in config.SAMPLES (default: ACTIVE_SAMPLE)")
    parser.add_argument("--tile", required=True,
                        help="tile folder name, e.g. 3_3")
    parser.add_argument("--out", default=None,
                        help="figure path (default: inside the tile folder)")
    parser.add_argument("--no-figure", action="store_true",
                        help="only print the summaries")
    args = parser.parse_args()

    cfg = sample(args.sample)
    he_path = Path(cfg["wsi_root"]) / args.tile / "5120_to256.npy"
    hr_path = Path(cfg["hrst_root"]) / args.tile / "HR_ST_256.npz"
    hr128_path = Path(cfg["hrst_root"]) / args.tile / "HR_ST_128.npz"
    spot_path = Path(cfg["spotst_root"]) / args.tile / "spot_ST.npz"

    print(f"tile {args.tile} of {cfg['name']}")
    describe("HR_ST_256", hr_path, config.HR_ST_256_SIDE)
    describe("HR_ST_128", hr128_path, config.HR_ST_128_SIDE)
    describe("spot_ST", spot_path, config.SPOT_BINS)

    if args.no_figure:
        return

    if not he_path.exists():
        print(f"  no H&E thumbnail at {he_path}, skipping the figure")
        return

    he = np.load(he_path)
    dense = load_npz(hr_path).toarray()
    mean_expression = dense.reshape(config.HR_ST_256_SIDE,
                                    config.HR_ST_256_SIDE, -1).mean(axis=2)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("H&E")
    axes[0].imshow(he if he.ndim == 3 else he, cmap=None if he.ndim == 3 else "gray")
    axes[1].set_title("mean expression, 10 um bins")
    axes[1].imshow(cv2.resize(normalise(mean_expression),
                              (config.HR_ST_256_SIDE, config.HR_ST_256_SIDE)),
                   cmap="viridis")
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout()

    out_path = Path(args.out) if args.out else he_path.parent / "qc_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"  figure written to {out_path}")


if __name__ == "__main__":
    main()
