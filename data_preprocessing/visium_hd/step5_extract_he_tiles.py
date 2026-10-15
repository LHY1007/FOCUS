"""Step 5 -- cut the H&E whole-slide image into FOCUS patches.

For every tile on the overlapping grid this writes three things into
``WSI/extract/{layer}/{i}_{j}/``:

    5120.npy         (5120, 5120, 3) uint8   the tile at 0.5 um/px
    5120_to256.npy   (256, 256, 3)   uint8   the same tile at 10 um/px
    320_to16.npy     (256, 16, 16, 3) uint8  the multi-scale sub-patch stack
    WSI_tile.png                             preview of the tile

The sub-patch stack is built by splitting the tile into 320 x 320 windows,
keeping the windows that look like real H&E tissue (enough texture, not
over-exposed, not background), downsampling each to 16 x 16, repeating the kept
list until it has 256 entries and sorting the result by complexity score. That
fixed-size stack is what the dataloader reads as the multi-scale H&E input.

Only the region covered by the transcript raster is tiled: the 2 um raster's
bounding box is projected through the registration matrix and tiles are laid out
inside it. A whole slide is tens of thousands of pixels wide and most of it is
glass.

    python step5_extract_he_tiles.py
    python step5_extract_he_tiles.py --workers 8 --limit 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pyvips
import spatialdata as sd

import config
from config import sample
from utils import (
    build_tile_grid,
    combined_affine,
    complexity_score,
    ensure_dir,
    project_extent,
    vips_tile_to_numpy,
)


# ---------------------------------------------------------------------------
# Sub-patch selection
# ---------------------------------------------------------------------------


def select_sub_patches(tile: np.ndarray, sub_pixel: int, small_side: int,
                       thresholds: dict) -> np.ndarray | None:
    """Return the ``(N, small_side, small_side, 3)`` stack of accepted sub-patches.

    *tile* is ``(H, W, 3)`` in RGB. A window is accepted when its share of
    near-flat pixels is below ``FLAT_RATIO``, its mean luminance is below
    ``BRIGHT_THRESH`` and its mean red channel is above ``R_AVG_THRESH``. Returns
    ``None`` when nothing passes, in which case the caller keeps the tile but
    writes no sub-patch stack.
    """
    height, width, _ = tile.shape
    n_rows = height // sub_pixel
    n_cols = width // sub_pixel

    windows = tile.reshape(n_rows, sub_pixel, n_cols, sub_pixel, 3).swapaxes(1, 2)

    red = windows[..., 0].astype(np.float32)
    green = windows[..., 1].astype(np.float32)
    blue = windows[..., 2].astype(np.float32)

    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    mean_luminance = luminance.mean(axis=(2, 3))
    mean_red = red.mean(axis=(2, 3))

    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    magnitude = cv2.addWeighted(cv2.convertScaleAbs(dx), 0.5,
                                cv2.convertScaleAbs(dy), 0.5, 0)
    flat = magnitude.reshape(n_rows, sub_pixel, n_cols, sub_pixel).swapaxes(1, 2)
    flat_ratio = (flat <= thresholds["grad"]).mean(axis=(2, 3))

    accepted = (
        (flat_ratio < thresholds["flat_ratio"])
        & (mean_luminance < thresholds["bright"])
        & (mean_red > thresholds["red"])
    )

    indices = np.argwhere(accepted)
    if indices.size == 0:
        return None

    patches = []
    for row, col in indices:
        window = windows[row, col]
        patches.append(cv2.resize(window, (small_side, small_side),
                                  interpolation=cv2.INTER_AREA))
    return np.stack(patches, axis=0)


def tile_sub_patch_stack(tile: np.ndarray, thresholds: dict) -> np.ndarray | None:
    """Build the 256-entry ``320_to16.npy`` stack for *tile*.

    The accepted sub-patches are cycled until the stack has
    :data:`config.PATCH_GRID` entries, then ordered by ascending complexity score
    so that the low-complexity views come first. Returns ``None`` when no
    sub-patch passed the tissue filter.
    """
    patches = select_sub_patches(tile, config.SUB_CROP_PIXEL, config.SMALL_SIDE,
                                 thresholds)
    if patches is None:
        return None

    repeats, remainder = divmod(config.PATCH_GRID, len(patches))
    stacked = list(patches) * repeats + list(patches[:remainder])
    stacked = np.stack(stacked, axis=0)

    scores = np.array([
        complexity_score(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))
        for patch in stacked
    ])
    return stacked[np.argsort(scores)]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def process_tile(job: dict) -> tuple[str, str]:
    """Cut and write one tile. Returns ``(tile_id, status)``.

    Runs in a ``multiprocessing`` worker, so it re-opens the slide handle rather
    than sharing one, and it takes every path as an argument.
    """
    tile_id = job["tile_id"]
    tile_dir = Path(job["wsi_root"]) / tile_id

    try:
        if (tile_dir / "5120.npy").exists():
            return tile_id, "exists"

        ensure_dir(tile_dir)

        slide = pyvips.Image.new_from_file(job["image_path"], access="sequential")
        scaled = slide.resize(job["scale"], kernel="linear")
        region = scaled.crop(job["x"], job["y"],
                             config.CROP_WSI_PIXEL, config.CROP_WSI_PIXEL)
        tile = vips_tile_to_numpy(region)

        thumbnail = cv2.resize(
            tile,
            (config.CROP_WSI_PIXEL // config.RESIZE_TIME,
             config.CROP_WSI_PIXEL // config.RESIZE_TIME),
            interpolation=cv2.INTER_AREA,
        )
        stack = tile_sub_patch_stack(tile, job["thresholds"])

        writes = [
            (tile_dir / "5120.npy", tile),
            (tile_dir / "5120_to256.npy", thumbnail),
        ]
        if stack is not None:
            writes.append((tile_dir / "320_to16.npy", stack))

        if job["write_async"]:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(np.save, str(path), array)
                           for path, array in writes]
                futures.append(pool.submit(
                    cv2.imwrite,
                    str(tile_dir / "WSI_tile.png"),
                    cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                ))
                for future in as_completed(futures):
                    future.result()
        else:
            for path, array in writes:
                np.save(str(path), array)
            cv2.imwrite(str(tile_dir / "WSI_tile.png"),
                        cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

        del tile, thumbnail, stack, region, scaled, slide
        gc.collect()
        return tile_id, "ok"

    except Exception as exc:                       # noqa: BLE001 - reported per tile
        return tile_id, f"error: {exc}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def read_mpp(raw_dir: Path) -> float:
    """Read ``microns_per_pixel`` from the sample's scalefactors file."""
    candidates = [
        raw_dir / "scalefactors_json.json",
        raw_dir / "spatial" / "scalefactors_json.json",
        raw_dir / "square_002um" / "spatial" / "scalefactors_json.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as handle:
                return float(json.load(handle)["microns_per_pixel"])
    raise FileNotFoundError(
        f"no scalefactors_json.json under {raw_dir}; "
        "expected one of " + ", ".join(str(p) for p in candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=None,
                        help="key in config.SAMPLES (default: ACTIVE_SAMPLE)")
    parser.add_argument("--workers", type=int, default=min(12, cpu_count()),
                        help="number of tile workers")
    parser.add_argument("--threads", action="store_true",
                        help="use threads instead of processes")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N tiles (for a trial run)")
    args = parser.parse_args()

    cfg = sample(args.sample)
    image_path = str(Path(cfg["raw_dir"]) / cfg["tissue_image"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    mpp = read_mpp(Path(cfg["raw_dir"]))
    scale = mpp / config.CROP_MPP

    slide = pyvips.Image.new_from_file(image_path, access="sequential")
    scaled = slide.resize(scale, kernel="linear")
    full_width, full_height = scaled.width, scaled.height
    print(f"slide at {config.CROP_MPP} um/px: {full_width} x {full_height}")
    del slide, scaled

    raster = sd.read_zarr(cfg["raster_zarr"])
    reference = raster["rasterized_002um"]
    st_height = int(reference.sizes["y"])
    st_width = int(reference.sizes["x"])
    del raster

    affine = combined_affine(cfg["affine"], scale)
    margin = config.CROP_WSI_PIXEL // 4
    x_min, y_min, x_max, y_max = project_extent(
        affine, st_width, st_height, margin, full_width, full_height
    )
    if x_min >= x_max or y_min >= y_max:
        raise RuntimeError(
            "the transcript raster does not overlap the slide; check the "
            "affine matrix and the microns_per_pixel value"
        )
    print(f"transcript extent at {config.CROP_MPP} um/px: "
          f"x[{x_min},{x_max}) y[{y_min},{y_max})")

    grid = build_tile_grid(x_min, y_min, x_max, y_max,
                           full_width, full_height,
                           config.TILE_STEP, config.CROP_WSI_PIXEL)
    if args.limit is not None:
        grid = grid[:args.limit]
    print(f"{len(grid)} tiles to cut, {args.workers} workers")

    thresholds = {
        "grad": config.GRAD_THRESH,
        "flat_ratio": config.FLAT_RATIO,
        "bright": config.BRIGHT_THRESH,
        "red": config.R_AVG_THRESH,
    }
    jobs = [
        {
            "tile_id": f"{entry['i']}_{entry['j']}",
            "x": entry["x"],
            "y": entry["y"],
            "image_path": image_path,
            "scale": scale,
            "wsi_root": str(cfg["wsi_root"]),
            "thresholds": thresholds,
            "write_async": True,
        }
        for entry in grid
    ]

    ensure_dir(cfg["wsi_root"])
    failures = 0
    if args.threads:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_tile, job) for job in jobs]
            for future in as_completed(futures):
                tile_id, status = future.result()
                if status not in ("ok", "exists"):
                    failures += 1
                    print(f"[{tile_id}] {status}")
    else:
        with Pool(processes=args.workers) as pool:
            for tile_id, status in pool.imap_unordered(process_tile, jobs, chunksize=2):
                if status not in ("ok", "exists"):
                    failures += 1
                    print(f"[{tile_id}] {status}")

    print(f"done, {failures} tiles reported a problem")
    print(f"tiles are under {cfg['wsi_root']}")


if __name__ == "__main__":
    main()
