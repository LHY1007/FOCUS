"""Step 3 -- read a Visium HD sample into a SpatialData Zarr store.

The raw Space Ranger output is a stack of text/parquet files plus a pyramidal
tissue image. ``spatialdata_io.visium_hd`` turns all of it into one in-memory
``SpatialData`` object that can be queried and plotted; this step persists that
object so the later steps do not have to re-parse the bundle.

    python step3_build_zarr.py
    python step3_build_zarr.py --then read     # print the store after writing
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import spatialdata as sd
from spatialdata_io import visium_hd

from config import sample


def build(cfg: dict, load_all_images: bool = True) -> Path:
    """Build and write the Zarr store for *cfg*; return its path."""
    raw_dir = Path(cfg["raw_dir"])
    zarr_path = Path(cfg["zarr_path"])
    tissue_image = cfg["tissue_image"]

    sdata = visium_hd(
        raw_dir,
        load_all_images=load_all_images,
        fullres_image_file=tissue_image,
    )

    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    sdata.write(zarr_path)
    return zarr_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=None,
                        help="key in config.SAMPLES (default: ACTIVE_SAMPLE)")
    parser.add_argument("--no-images", action="store_true",
                        help="skip loading the full-resolution images")
    parser.add_argument("--then", choices=["none", "read"],
                        help="'read' re-opens the store and prints its summary")
    args = parser.parse_args()

    cfg = sample(args.sample)
    zarr_path = build(cfg, load_all_images=not args.no_images)
    print(f"wrote {zarr_path}")

    if args.then == "read":
        sdata = sd.SpatialData.read(zarr_path)
        print(sdata)

    print(f'view with "python -m napari_spatialdata view {zarr_path}"')


if __name__ == "__main__":
    main()
