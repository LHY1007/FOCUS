"""Step 1 -- download a Visium HD bundle from the 10x Genomics CDN.

The 10x sample pages expose five files per Visium HD dataset: the Cloupe bundle,
the feature-slice matrix, the spatial folder, the binned outputs and the tissue
image. This step fetches all five into ``raw_dir``, unpacks the two archives and
flattens ``binned_outputs/square_*um`` up one level, which is the layout
``step3_build_zarr.py`` expects.

Run it once per sample; every later step works on the unpacked directory.

    python step1_download.py
    python step1_download.py --sample Human_Breast_Cancer_FFPE
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from config import sample
from utils import download

#: File suffixes 10x publishes for a Visium HD sample, in download order.
BUNDLE_SUFFIXES = [
    "_cloupe_008um.cloupe",
    "_feature_slice.h5",
    "_spatial.tar.gz",
    "_binned_outputs.tar.gz",
    "_tissue_image.tif",
]

#: The two members that are archives rather than plain files.
ARCHIVES = ["_spatial.tar.gz", "_binned_outputs.tar.gz"]

#: Base of the CDN path. The dataset name is appended, then the file suffix.
CDN_ROOT = "https://cf.10xgenomics.com/samples/spatial-exp"


def build_urls(dataset: str, version: str) -> list[str]:
    """Return the five download URLs for *dataset* (e.g. ``Visium_HD_FF_Human_Tonsil``)."""
    base = f"{CDN_ROOT}/{version}/{dataset}/{dataset}"
    return [base + suffix for suffix in BUNDLE_SUFFIXES]


def unpack(archive: Path, dest: Path) -> None:
    """Extract a ``.tar.gz`` into *dest* and remove it."""
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(dest)], check=True)
    archive.unlink()


def flatten_binned_outputs(data_dir: Path) -> None:
    """Move ``binned_outputs/square_*um`` up into *data_dir* and drop the wrapper.

    Space Ranger nests the per-bin matrices one level down. The rest of the
    pipeline reads ``raw_dir/square_002um`` directly, so the wrapper is removed
    once its children have been moved.
    """
    binned = data_dir / "binned_outputs"
    if not binned.is_dir():
        return
    for child in binned.iterdir():
        target = data_dir / child.name
        if not target.exists():
            shutil.move(str(child), str(target))
    shutil.rmtree(binned)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=None,
                        help="key in config.SAMPLES (default: ACTIVE_SAMPLE)")
    parser.add_argument("--dataset", default=None,
                        help="10x dataset name, e.g. Visium_HD_FF_Human_Tonsil")
    parser.add_argument("--version", default="3.1.2",
                        help="10x spatial-exp release the dataset belongs to")
    args = parser.parse_args()

    cfg = sample(args.sample)
    data_dir = Path(cfg["raw_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset or cfg["name"]
    urls = build_urls(dataset, args.version)

    print(f"downloading {dataset} into {data_dir}")
    for url in urls:
        name = Path(url).name
        target = data_dir / name
        if target.exists():
            print(f"  exists   {name}")
            continue
        print(f"  fetching {name}")
        download(url, target, name)

    for suffix in ARCHIVES:
        archive = next(data_dir.glob(f"*{suffix}"), None)
        if archive is not None:
            unpack(archive, data_dir)

    flatten_binned_outputs(data_dir)

    print("contents:")
    for entry in sorted(os.listdir(data_dir)):
        print(f"  {entry}")


if __name__ == "__main__":
    main()
