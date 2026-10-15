"""Step 6 -- bin the transcripts into the HR_ST and spot_ST grids.

For every tile written by ``step5_extract_he_tiles.py`` this reads each gene out
of the rasterised Zarr as a 2 um-resolution image, warps it into the tile frame
with the registration matrix, block-sums it to the three output scales and saves
the result as a sparse matrix:

    HR_ST/extract/{layer}/{i}_{j}/HR_ST_256.npz   (65536, G)   10 um bins
    HR_ST/extract/{layer}/{i}_{j}/HR_ST_128.npz   (16384, G)   20 um bins
    spot_ST/extract/{layer}/{i}_{j}/spot_ST.npz   (676, G)     100 um bins

The three grids describe the same field of view at different resolutions. They
are what makes the Visium HD branch interchangeable with the Visium and Xenium
branches in the dataloader.

Genes are processed in batches so that only a slice of the raster is held in
memory at a time; the batch size is ``--gene-batch``.

    python step6_extract_st_grids.py
    python step6_extract_st_grids.py --workers 8 --gene-batch 512
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
import pandas as pd
import spatialdata as sd
from scipy.sparse import csr_matrix, save_npz

import config
from config import sample
from utils import (
    aggregate_block_sum,
    build_tile_grid,
    combined_affine,
    ensure_dir,
    project_extent,
)

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
    raise FileNotFoundError(f"no scalefactors_json.json under {raw_dir}")


def warp_gene(gene_image: np.ndarray, affine: np.ndarray, size: int,
              buffer: np.ndarray) -> np.ndarray:
    """Warp one gene image into the tile frame using a preallocated *buffer*.

    *gene_image* is the 2 um raster of a single gene; *affine* already has the
    tile origin subtracted. The result is written into *buffer* and returned.
    """
    if gene_image.dtype != np.float32:
        gene_image = gene_image.astype(np.float32)
    return cv2.warpAffine(
        gene_image, affine, (size, size),
        dst=buffer, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def process_tile(job: dict) -> tuple[str, str]:
    """Bin the transcripts for one tile. Returns ``(tile_id, status)``."""
    tile_id = job["tile_id"]
    hrst_dir = Path(job["hrst_root"]) / tile_id
    spot_dir = Path(job["spotst_root"]) / tile_id
    target = hrst_dir / "HR_ST_256.npz"

    try:
        if target.exists():
            return tile_id, "exists"

        ensure_dir(hrst_dir)
        ensure_dir(spot_dir)

        # Each worker opens its own handle: the raster store is not shareable
        # across processes.
        raster = sd.read_zarr(job["raster_zarr"])
        gene_names = job["gene_names"]
        n_genes = len(gene_names)

        # The tile's own origin is removed from the affine so the warp lands
        # inside a CROP_WSI_PIXEL-sized window.
        affine = np.array(job["affine"], dtype=np.float32)
        affine[0, 2] -= job["x"]
        affine[1, 2] -= job["y"]

        hr256 = np.zeros((config.HR_ST_256_SIDE ** 2, n_genes), dtype=np.float32)
        hr128 = np.zeros((config.HR_ST_128_SIDE ** 2, n_genes), dtype=np.float32)
        spot = np.zeros((config.SPOT_BINS ** 2, n_genes), dtype=np.float32)

        buffer = np.empty((config.CROP_WSI_PIXEL, config.CROP_WSI_PIXEL),
                          dtype=np.float32)

        # The spot grid needs a padded buffer: 5120 is not divisible by 200, so
        # the last row/column would otherwise be dropped.
        pad_h = config.SPOT_BINS * config.SPOT_RELA_WSI_MPP - config.CROP_WSI_PIXEL
        pad_w = pad_h
        spot_buffer = None
        if pad_h > 0 or pad_w > 0:
            spot_buffer = np.zeros(
                (config.CROP_WSI_PIXEL + pad_h, config.CROP_WSI_PIXEL + pad_w),
                dtype=np.float32,
            )

        batch = max(1, int(job["gene_batch"]))
        for start in range(0, n_genes, batch):
            stop = min(start + batch, n_genes)

            for offset, gene in enumerate(gene_names[start:stop]):
                column = start + offset

                gene_image = raster["rasterized_002um"].sel(c=gene).data.compute()
                if hasattr(gene_image, "values"):
                    gene_image = gene_image.values

                warped = warp_gene(gene_image, affine,
                                   config.CROP_WSI_PIXEL, buffer)
                as_3d = warped[:, :, None]

                hr256[:, column] = aggregate_block_sum(
                    as_3d, config.ST_RELA_WSI_MPP).reshape(-1)
                hr128[:, column] = aggregate_block_sum(
                    as_3d, config.ST_RELA_WSI_MPP * 2).reshape(-1)

                if spot_buffer is not None:
                    spot_buffer[:config.CROP_WSI_PIXEL,
                                :config.CROP_WSI_PIXEL] = warped
                    spot_input = spot_buffer
                else:
                    spot_input = warped
                spot[:, column] = aggregate_block_sum(
                    spot_input[:, :, None], config.SPOT_RELA_WSI_MPP).reshape(-1)

            gc.collect()

        save_npz(str(hrst_dir / "HR_ST_256.npz"), csr_matrix(hr256))
        save_npz(str(hrst_dir / "HR_ST_128.npz"), csr_matrix(hr128))
        save_npz(str(spot_dir / "spot_ST.npz"), csr_matrix(spot))

        del raster, hr256, hr128, spot, buffer, spot_buffer
        gc.collect()
        return tile_id, "ok"

    except Exception as exc:                       # noqa: BLE001 - reported per tile
        return tile_id, f"error: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=None,
                        help="key in config.SAMPLES (default: ACTIVE_SAMPLE)")
    parser.add_argument("--workers", type=int, default=min(12, cpu_count()),
                        help="number of tile workers")
    parser.add_argument("--threads", action="store_true",
                        help="use threads instead of processes")
    parser.add_argument("--gene-batch", type=int, default=1024,
                        help="how many genes to hold in memory at once")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N tiles (for a trial run)")
    args = parser.parse_args()

    cfg = sample(args.sample)
    image_path = Path(cfg["raw_dir"]) / cfg["tissue_image"]
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    scale = read_mpp(Path(cfg["raw_dir"])) / config.CROP_MPP

    slide = pyvips.Image.new_from_file(str(image_path), access="sequential")
    scaled = slide.resize(scale, kernel="linear")
    full_width, full_height = scaled.width, scaled.height
    del slide, scaled

    raster = sd.read_zarr(cfg["raster_zarr"])
    gene_names = list(raster["rasterized_002um"].coords["c"].values)
    st_height = int(raster["rasterized_002um"].sizes["y"])
    st_width = int(raster["rasterized_002um"].sizes["x"])
    del raster
    print(f"{len(gene_names)} genes to bin")

    affine = combined_affine(cfg["affine"], scale)
    margin = config.CROP_WSI_PIXEL // 4
    x_min, y_min, x_max, y_max = project_extent(
        affine, st_width, st_height, margin, full_width, full_height
    )
    grid = build_tile_grid(x_min, y_min, x_max, y_max,
                           full_width, full_height,
                           config.TILE_STEP, config.CROP_WSI_PIXEL)
    if args.limit is not None:
        grid = grid[:args.limit]
    print(f"{len(grid)} tiles to bin, {args.workers} workers")

    jobs = [
        {
            "tile_id": f"{entry['i']}_{entry['j']}",
            "x": entry["x"],
            "y": entry["y"],
            "affine": affine,
            "raster_zarr": cfg["raster_zarr"],
            "hrst_root": str(cfg["hrst_root"]),
            "spotst_root": str(cfg["spotst_root"]),
            "gene_names": gene_names,
            "gene_batch": args.gene_batch,
        }
        for entry in grid
    ]

    ensure_dir(cfg["hrst_root"])
    ensure_dir(cfg["spotst_root"])

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
            for tile_id, status in pool.imap_unordered(process_tile, jobs, chunksize=1):
                if status not in ("ok", "exists"):
                    failures += 1
                    print(f"[{tile_id}] {status}")

    # The gene column order is part of the contract with the dataloader, so it
    # is written next to the tiles.
    gene_csv = cfg["wsi_root"] / "gene_names.csv"
    ensure_dir(cfg["wsi_root"])
    pd.DataFrame(gene_names, columns=["Gene_Name"]).to_csv(gene_csv, index=False)
    print(f"gene order written to {gene_csv}")
    print(f"done, {failures} tiles reported a problem")


if __name__ == "__main__":
    main()
