"""Shared helpers for the Visium HD preprocessing pipeline.

Download and archive helpers support ``step1_download.py``. Tile and image
helpers support ``step5_extract_he_tiles.py`` and ``step6_extract_st_grids.py``.
Functions take explicit arguments for use in multiprocessing workers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


class TqdmDownload(tqdm):
    """``tqdm`` subclass that understands the ``urlretrieve`` progress callback."""

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.update({"unit": "B", "unit_scale": True, "unit_divisor": 1024})
        super().__init__(*args, **kwargs)

    def update_to(self, nblocks=1, blocksize=1, total=-1):
        self.total = total
        self.update(nblocks * blocksize - self.n)


def is_aria2c_installed() -> bool:
    """Return ``True`` when ``aria2c`` is on ``PATH``."""
    return subprocess.call(["which", "aria2c"]) == 0


def download(url: str, outfile, desc: str = "") -> None:
    """Download *url* to *outfile*.

    ``curl`` is used rather than ``urllib`` because the 10x CDN throttles a
    plain Python client noticeably; ``aria2c`` is preferred when available.
    """
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if is_aria2c_installed():
        subprocess.check_call(
            [
                "aria2c",
                "--allow-overwrite",
                "-x", "4",
                "--dir", str(outfile.parent),
                "-o", outfile.name,
                url,
            ]
        )
    else:
        subprocess.check_call(["curl", "-L", "-o", str(outfile), url])


def unzip(file, outdir=None, files=None, rm=True) -> None:
    """Extract *file* into *outdir*, optionally only the members in *files*."""
    outdir = Path(outdir) if outdir is not None else Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(file) as zfile:
        if files is not None:
            for member in files:
                zfile.extract(member, outdir)
        else:
            zfile.extractall(outdir)
    if rm:
        os.unlink(file)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def ensure_dir(path) -> Path:
    """Create *path* (and parents) if needed and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_all_files(path) -> None:
    """Delete the *contents* of directory *path*, keeping the directory itself.

    Only used to refresh a tile folder that is being recomputed; nested
    directories are removed recursively.
    """
    path = Path(path)
    if not path.is_dir():
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


# ---------------------------------------------------------------------------
# Whole-slide-image helpers
# ---------------------------------------------------------------------------


def vips_tile_to_numpy(image) -> np.ndarray:
    """Materialise a ``pyvips`` region as an ``(H, W, bands)`` uint8 array.

    ``pyvips`` keeps the whole slide on disk and only computes the requested
    region; this copies that region into RAM once so the rest of the tile
    processing can use NumPy.
    """
    mem = image.write_to_memory()
    h, w, bands = image.height, image.width, image.bands
    return np.ndarray(buffer=mem, dtype=np.uint8, shape=(h, w, bands))


def aggregate_block_sum(tile: np.ndarray, factor: int) -> np.ndarray:
    """Block-sum *tile* over ``factor x factor`` windows.

    ``tile`` is ``(H, W, C)``; the result is ``(H // factor, W // factor, C)``.
    The right/bottom remainder that does not fill a whole window is cropped, and
    the accumulator dtype is :data:`config.COUNT_DTYPE` (``int16``), which is the
    convention every ``HR_ST_*.npz`` / ``spot_ST.npz`` in the dataset follows.
    """
    height, width, channels = tile.shape
    new_h = height // factor
    new_w = width // factor
    cropped = tile[: new_h * factor, : new_w * factor, :]
    summed = cropped.reshape(new_h, factor, new_w, factor, channels).sum(axis=(1, 3))
    return summed.astype(np.int16)


def complexity_score(gray_uint8: np.ndarray) -> float:
    """Shannon entropy of the 256-bin intensity histogram of a grayscale patch.

    This is the "complexity score" the paper uses to order the sub-patches
    inside ``320_to16.npy``: a flat region has a low score, a region with many
    distinct intensities has a high one. ``skimage.measure.shannon_entropy``
    computes the same number; this version avoids the extra dependency and the
    per-call histogram construction.
    """
    hist = cv2.calcHist([gray_uint8], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(hist.sum(), 1.0)
    nonzero = prob[prob > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of *image* as a uint8 array.

    Input is expected in OpenCV channel order (BGR). The two directional
    responses are scaled to 8 bit and averaged, which is the classic
    ``getGradientMagnitude`` from the stLearn/Visium tooling.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return cv2.addWeighted(cv2.convertScaleAbs(dx), 0.5,
                           cv2.convertScaleAbs(dy), 0.5, 0)


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------


def combined_affine(affine_2x3, scale: float) -> np.ndarray:
    """Compose a 2 x 3 registration matrix with an isotropic *scale*.

    The stored registration maps rasterised bin coordinates onto the H&E frame
    at its native resolution; *scale* brings it to the resolution the tiles are
    cut at (``MPP / CROP_MPP``).
    """
    affine = np.asarray(affine_2x3, dtype=np.float64)
    scaling = np.array([[scale, 0.0, 0.0],
                        [0.0, scale, 0.0],
                        [0.0, 0.0, 1.0]])
    combined = scaling @ np.vstack([affine, [0.0, 0.0, 1.0]])
    return combined[:2, :]


def project_extent(affine_2x3, width: int, height: int, margin: int,
                   limit_w: int, limit_h: int) -> tuple[int, int, int, int]:
    """Project the four corners of a ``width x height`` grid through *affine_2x3*.

    Returns the axis-aligned bounding box ``(x_min, y_min, x_max, y_max)`` of the
    projection, grown by *margin* pixels and clipped to ``limit_w x limit_h``.
    Used to restrict tiling to the region the transcripts actually cover, which
    keeps a 60 000 px slide from producing tens of thousands of empty tiles.
    """
    corners = np.array([
        [0, 0, 1],
        [width - 1, 0, 1],
        [0, height - 1, 1],
        [width - 1, height - 1, 1],
    ], dtype=np.float64)

    full = np.vstack([np.asarray(affine_2x3, dtype=np.float64), [0, 0, 1]])
    projected = (full @ corners.T).T

    xs, ys = projected[:, 0], projected[:, 1]
    x_min = max(0, int(np.floor(xs.min()) - margin))
    y_min = max(0, int(np.floor(ys.min()) - margin))
    x_max = min(limit_w, int(np.ceil(xs.max()) + margin))
    y_max = min(limit_h, int(np.ceil(ys.max()) + margin))
    return x_min, y_min, x_max, y_max


def build_tile_grid(x_min: int, y_min: int, x_max: int, y_max: int,
                    w_full: int, h_full: int, tile_step: int,
                    crop_pixel: int) -> list[dict]:
    """Enumerate the tiles covering ``[x_min, x_max) x [y_min, y_max)``.

    Tiles are anchored on *tile_step* and advance by half a tile, so neighbours
    overlap by 50 %. Each entry records both the top-left pixel of the tile in
    the full slide and its index inside the covered region; the index is what
    names the output folder (``{i}_{j}``), the pixel is what gets cropped.
    """
    x_start = max(0, (x_min // tile_step) * tile_step)
    y_start = max(0, (y_min // tile_step) * tile_step)
    x_end = min(x_max, w_full)
    y_end = min(y_max, h_full)

    grid = []
    for x in range(x_start, max(x_start, x_end - crop_pixel) + 1, tile_step):
        for y in range(y_start, max(y_start, y_end - crop_pixel) + 1, tile_step):
            grid.append({
                "x": x,
                "y": y,
                "i": (x - x_start) // tile_step,
                "j": (y - y_start) // tile_step,
            })
    return grid

