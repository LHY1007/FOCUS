"""Step 4 -- rasterise the binned tables into gene x (y, x) image-like arrays.

``rasterize_bins`` reshapes a ``square_*um`` table into a dense raster laid out
on the Visium HD array grid. The result is what makes the transcript data
indexable like an image, which is how ``step6_extract_st_grids.py`` can read a
single gene as a 2 um-resolution picture and warp it into a tile.

The step also makes the gene coordinate unique. Visium HD panels contain
duplicated symbols, and the raster coordinate ``c`` has to be unique for
``.sel(c=...)`` to work; the second and later copies get a ``_1``, ``_2``
suffix, in the order they appear.

Output is a second Zarr store (``raster_zarr``); the input store is left
untouched.

    python step4_rasterize_bins.py
"""

from __future__ import annotations

import argparse
from collections import Counter

import spatialdata as sd
from spatialdata import rasterize_bins

from config import sample

#: Bin sizes to rasterise, largest first so the 2 um raster is the reference.
BIN_SIZES = ["016", "008", "002"]

#: The bin whose coordinate order defines the gene columns everywhere downstream.
REFERENCE_BIN = "002"


def make_gene_names_unique(gene_names: list[str]) -> list[str]:
    """Return *gene_names* with repeated symbols disambiguated by a numeric suffix.

    The first occurrence keeps its name; later ones become ``{name}_1``,
    ``{name}_2``, and so on. Order is preserved, so the result stays aligned with
    the raster's ``c`` coordinate.
    """
    counts = Counter(gene_names)
    seen: dict[str, int] = {}
    unique = []
    for name in gene_names:
        if counts[name] > 1:
            index = seen.get(name, 0)
            unique.append(name if index == 0 else f"{name}_{index}")
            seen[name] = index + 1
        else:
            unique.append(name)
    return unique


def find_bin_element(sdata: sd.SpatialData, bin_size: str) -> str:
    """Return the name of the element holding the ``square_{bin}um`` data.

    ``spatialdata_io.visium_hd`` prefixes the element with the sample name, which
    is not something this pipeline knows, so the element is looked up by suffix
    rather than reconstructed.
    """
    suffix = f"square_{bin_size}um"
    for collection in (sdata.images, sdata.shapes):
        for name in collection:
            if name.endswith(suffix):
                return name
    raise KeyError(f"no element ending in {suffix!r} in the store")


def rasterize(sdata: sd.SpatialData, bin_sizes: list[str]) -> list[str]:
    """Add one ``rasterized_{bin}um`` element per bin size; return the duplicates."""
    for bin_size in bin_sizes:
        table = sdata.tables[f"square_{bin_size}um"]
        # rasterize_bins() walks the table column-wise, which is only cheap on CSC.
        table.X = table.X.tocsc()

        sdata[f"rasterized_{bin_size}um"] = rasterize_bins(
            sdata, find_bin_element(sdata, bin_size), f"square_{bin_size}um",
            "array_col", "array_row",
        )

    raster = sdata[f"rasterized_{REFERENCE_BIN}um"]
    gene_names = list(raster.coords["c"].values)
    duplicates = [name for name, count in Counter(gene_names).items() if count > 1]
    raster.coords["c"] = make_gene_names_unique(gene_names)
    return duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=None,
                        help="key in config.SAMPLES (default: ACTIVE_SAMPLE)")
    parser.add_argument("--bins", nargs="*", default=BIN_SIZES,
                        help="bin sizes to rasterise")
    args = parser.parse_args()

    cfg = sample(args.sample)
    sdata = sd.read_zarr(cfg["zarr_path"])

    duplicates = rasterize(sdata, args.bins)
    if duplicates:
        print(f"disambiguated {len(duplicates)} duplicated gene symbols")

    gene_names = list(sdata[f"rasterized_{REFERENCE_BIN}um"].coords["c"].values)
    print(f"{len(gene_names)} genes in the {REFERENCE_BIN}um raster")

    out_path = cfg["raster_zarr"]
    sdata.write(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
