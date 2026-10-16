"""Nucleus and cell segmentation of the H&E patches.

Nuclei are segmented with the pretrained StarDist2D H&E model, small detections
are dropped, and the nuclei are then grown isotropically into a full-cell
partition that does not overlap. The result is written next to each patch as
``cell_mask.npy``, shaped ``(256, 256, N)`` -- one binary channel per cell on the
same 10 um/px grid as ``5120_to256.npy``. ``guided_diffusion/load_dataset.py``
loads it with ``transpose(2, 0, 1)`` and ``UNetModel`` reduces it to
``[N, 256, 256]``, so the channel axis has to be the cell axis.

The StarDist call follows the Visium HD recipe published by 10x Genomics (see
``Methods/NucleiSegmentation.py`` in 10XGenomics/HumanColonCancer_VisiumHD):
``2D_versatile_he``, percentile normalisation, and ``predict_instances_big`` with
a block size, a guaranteed overlap and a discarded context so that a whole
section is segmented in blocks and stitched back without duplicating objects at
the seams. The segmentation thresholds are left at the model defaults; the 10x
script overrides them to 0.01 / 0.001 for its dense colon sections, which is a
per-tissue choice, not part of the recipe.

The expansion is implemented here rather than delegated to
``skimage.segmentation.expand_labels``: it is the one step whose guarantee
matters, so it is worth having in the repository where it can be read.

    python stardist_segmentation.py --all --root /path/to/WSI/extract/20240612
    python stardist_segmentation.py --tile 3_3 --root ...
    python stardist_segmentation.py --slide /path/to/section.tif --out labels.npy
"""

from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Pretrained StarDist model. ``2D_versatile_he`` is the brightfield H&E model;
#: the fluorescence models are trained on a different modality and are not
#: interchangeable with it.
MODEL_NAME = "2D_versatile_he"

#: Segmentation thresholds. ``None`` keeps the model's own values (probability
#: 0.5, NMS overlap 0.4).
PROB_THRESH = None
NMS_THRESH = None

#: Percentile normalisation applied before prediction, then the block geometry
#: for ``predict_instances_big``. Every object must be smaller than
#: ``block_overlap``, and ``block_overlap + 2 * block_context < block_size``.
NORM_LO = 1.0
NORM_HI = 99.8
BLOCK_SIZE = 4096
BLOCK_OVERLAP = 128
BLOCK_CONTEXT = 128

#: Detections smaller than this many pixels are debris. At 0.5 um/px, 100 px is a
#: 5 x 5 um object, smaller than any real nucleus.
MIN_NUCLEUS_AREA = 100

#: Distance the nuclei are grown by, in micrometres. This is the cytoplasm the
#: nuclear stain does not cover; 5 um is a typical epithelial nucleus-to-cell
#: margin and is the one number that should be re-checked per tissue.
EXPAND_UM = 5.0

#: Physical pixel size of the patches handed to the segmenter.
PATCH_MPP = 0.5

#: Tile geometry, matching the rest of the preprocessing pipeline.
TILE_SIDE = 5120
MASK_SIDE = 256
DOWNSAMPLE = TILE_SIDE // MASK_SIDE          # 20 px at 0.5 um/px = 10 um

#: A 10 um/px pixel goes to the cell covering at least this share of the
#: 20 x 20 block it summarises; below it the pixel stays background.
MIN_COVERAGE = 0.25

#: Tissue mask: Otsu threshold, then close to fill holes and open to drop specks.
CLOSE_RADIUS = 10
OPEN_RADIUS = 5

#: A patch with less tissue than this is skipped without running the detector.
MIN_TISSUE_COVERAGE = 0.05


# ---------------------------------------------------------------------------
# Tissue mask
# ---------------------------------------------------------------------------


def disk(radius: int) -> np.ndarray:
    """Return a circular structuring element of the given radius."""
    size = 2 * radius + 1
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ((xx ** 2 + yy ** 2) <= radius ** 2).astype(np.uint8)


def tissue_mask(rgb: np.ndarray, close_radius: int = CLOSE_RADIUS,
                open_radius: int = OPEN_RADIUS) -> np.ndarray:
    """Boolean mask of the tissue-covered area of *rgb*.

    Background glass is bright and near-uniform, so Otsu separates it from
    stained tissue without a fixed intensity. Closing fills the holes inside
    tissue and opening drops isolated specks.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, disk(close_radius))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, disk(open_radius))
    return opened.astype(bool)


# ---------------------------------------------------------------------------
# Nuclei
# ---------------------------------------------------------------------------


def load_model(model_name: str = MODEL_NAME):
    """Load the pretrained StarDist2D model."""
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained(model_name)


def normalise(rgb: np.ndarray, low: float = NORM_LO, high: float = NORM_HI) -> np.ndarray:
    """Percentile-normalise *rgb* the way the pretrained models expect.

    ``axis=(0, 1, 2)`` normalises the three H&E channels jointly, which is the
    normalisation the model was trained under; normalising each channel on its
    own shifts the stain ratios the detector keys on.
    """
    from csbdeep.utils import normalize

    return normalize(rgb, low, high, axis=(0, 1, 2))


def segment_patch(model, rgb: np.ndarray, prob_thresh=None,
                  nms_thresh=None) -> np.ndarray:
    """Segment a single patch that fits in memory; return an int32 label image."""
    kwargs = {}
    if prob_thresh is not None:
        kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        kwargs["nms_thresh"] = nms_thresh

    labels, _polys = model.predict_instances(normalise(rgb), **kwargs)
    return labels.astype(np.int32, copy=False)


def segment_section(model, rgb: np.ndarray, block_size: int = BLOCK_SIZE,
                    min_overlap: int = BLOCK_OVERLAP,
                    context: int = BLOCK_CONTEXT,
                    n_tiles: tuple | None = None) -> tuple[np.ndarray, dict]:
    """Segment a whole section in blocks and stitch the nuclei back together.

    ``predict_instances_big`` splits the image, predicts each block, discards the
    context ring and relabels the survivors so the stitched result matches a
    single-pass prediction. The assumption it relies on is that no object is
    larger than *min_overlap*; objects that violate it are split at the seam and
    reported in the polygon dictionary rather than silently kept.

    Returns the label image and the polygon dictionary StarDist produces, which
    carries the object outlines the rest of the project uses.
    """
    labels, polys = model.predict_instances_big(
        normalise(rgb),
        axes="YXC",
        block_size=block_size,
        min_overlap=min_overlap,
        context=context,
        prob_thresh=PROB_THRESH,
        nms_thresh=NMS_THRESH,
        normalizer=None,
        n_tiles=n_tiles,
    )
    return labels.astype(np.int32, copy=False), polys


def drop_small_objects(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove labels covering fewer than *min_area* pixels.

    Areas are counted per label, not on the binarised mask, so a sheet of
    touching nuclei is not mistaken for a single object.
    """
    if labels.size == 0:
        return labels
    counts = np.bincount(labels.ravel())
    keep = np.zeros(counts.size, dtype=bool)
    keep[1:] = counts[1:] >= min_area
    out = labels.copy()
    out[~keep[labels]] = 0
    return out


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def expand_non_overlapping(labels: np.ndarray, distance_um: float,
                           mpp: float = PATCH_MPP) -> np.ndarray:
    """Grow every label isotropically into the background without overlapping.

    Each background pixel inside the reach of some nucleus is assigned to the
    nucleus **nearest to it**, measured as a Euclidean distance in micrometres.
    Because "nearest" picks exactly one nucleus per pixel, no pixel can end up
    claimed by two labels, so the result is a partition of the tissue and the
    channels built from it are disjoint by construction. That is the property a
    per-cell weighting needs; a plain morphological dilation does not have it,
    because two expanding labels can both write the same pixel and the last
    writer silently wins.

    Ties are resolved by the distance transform's own ordering, which is stable
    for a given input. Pixels further than *distance_um* from any nucleus stay
    background.

    Parameters
    ----------
    labels:
        Integer label image, 0 for background.
    distance_um:
        How far to grow, in micrometres. ``0`` returns *labels* unchanged.
    mpp:
        Micrometres per pixel of *labels*; used so that the distance is isotropic
        in physical units rather than in pixels.
    """
    if distance_um <= 0:
        return labels

    background = labels == 0
    if not background.any():
        return labels

    # Distance from every background pixel to the nearest labelled pixel, plus
    # the coordinates of that pixel. `background` is 1 on the background, so its
    # zeros are exactly the labelled pixels the transform measures against.
    distances, nearest = ndi.distance_transform_edt(
        background, sampling=(mpp, mpp), return_indices=True
    )
    nearest_labels = labels[tuple(nearest)]

    grown = labels.copy()
    reachable = background & (distances <= distance_um)
    grown[reachable] = nearest_labels[reachable]
    return grown


# ---------------------------------------------------------------------------
# Cell masks
# ---------------------------------------------------------------------------


def downsample_labels(labels: np.ndarray, out_side: int = MASK_SIDE,
                      factor: int = DOWNSAMPLE,
                      min_coverage: float = MIN_COVERAGE) -> np.ndarray:
    """Reduce a label image to *out_side* x *out_side*, one pixel per 10 um.

    Each output pixel takes the label covering most of the ``factor x factor``
    block it summarises, counting only labelled pixels; a block whose labelled
    share is below *min_coverage* stays background, so a cell that merely clips a
    block is not promoted into a pixel it does not occupy.
    """
    height, width = labels.shape
    usable_h = min(out_side, height // factor)
    usable_w = min(out_side, width // factor)
    trimmed = labels[:usable_h * factor, :usable_w * factor]

    blocks = (trimmed.reshape(usable_h, factor, usable_w, factor)
              .transpose(0, 2, 1, 3)
              .reshape(usable_h, usable_w, factor * factor))

    coverage = (blocks > 0).mean(axis=2)

    out = np.zeros((out_side, out_side), dtype=labels.dtype)
    flat = blocks.reshape(-1, factor * factor)
    rows, cols = np.divmod(np.arange(flat.shape[0]), usable_w)

    for index in range(flat.shape[0]):
        values = flat[index]
        nonzero = values[values > 0]
        if nonzero.size:
            out[rows[index], cols[index]] = np.bincount(nonzero).argmax()

    # Only the region the input actually covered was written; a patch smaller
    # than out_side * factor leaves the remainder at zero, so the coverage mask
    # has to be applied to that same corner rather than to the whole array.
    corner = out[:usable_h, :usable_w]
    corner[coverage < min_coverage] = 0
    return out


def stack_cells(labels: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Turn a 2-D label image into ``(H, W, N)`` binary per-cell channels.

    A patch with no cells yields ``(MASK_SIDE, MASK_SIDE, 0)`` so the channel
    axis is always present and callers never have to special-case it.
    """
    ids = np.unique(labels)
    ids = ids[ids > 0]
    height, width = labels.shape

    channels = np.zeros((height, width, ids.size), dtype=dtype)
    for index, label in enumerate(ids):
        channels[..., index] = labels == label
    return channels


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def cell_mask_for_patch(model, rgb: np.ndarray, expand_um: float,
                        min_area: int) -> np.ndarray:
    """Run the whole recipe on one patch and return ``(256, 256, N)``."""
    nuclei = segment_patch(model, rgb, PROB_THRESH, NMS_THRESH)
    nuclei = drop_small_objects(nuclei, min_area)
    cells = expand_non_overlapping(nuclei, expand_um, PATCH_MPP)
    return stack_cells(downsample_labels(cells))


def process_tile(model, patch_dir: Path, expand_um: float, min_area: int,
                 overwrite: bool = False) -> tuple[str, int]:
    """Write ``cell_mask.npy`` for one patch directory.

    Returns a status string and the number of cells written.
    """
    source = patch_dir / "5120.npy"
    target = patch_dir / "cell_mask.npy"

    if target.exists() and not overwrite:
        return "exists", 0
    if not source.exists():
        return f"no {source.name}", 0

    rgb = np.load(source)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    rgb = np.ascontiguousarray(rgb[..., :3])

    if tissue_mask(rgb).mean() < MIN_TISSUE_COVERAGE:
        return "background", 0

    channels = cell_mask_for_patch(model, rgb, expand_um, min_area)
    np.save(target, channels)

    count = channels.shape[2]
    del rgb, channels
    gc.collect()
    return "ok", count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None,
                        help="WSI/extract/{layer} holding the {i}_{j} patch folders")
    parser.add_argument("--tile", default=None,
                        help="one patch folder to process, e.g. 3_3")
    parser.add_argument("--all", action="store_true",
                        help="process every patch folder under --root")
    parser.add_argument("--slide", default=None,
                        help="segment a whole section in one pass")
    parser.add_argument("--out", default=None,
                        help="output path for --slide (label image)")
    parser.add_argument("--polys", default=None,
                        help="output path for the --slide polygon pickle")
    parser.add_argument("--distance-um", type=float, default=EXPAND_UM,
                        help="how far to grow each nucleus, in micrometres")
    parser.add_argument("--mpp", type=float, default=PATCH_MPP,
                        help="micrometres per pixel of the input")
    parser.add_argument("--min-area", type=int, default=MIN_NUCLEUS_AREA,
                        help="drop detections smaller than this many pixels")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="StarDist2D pretrained model name")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--min-overlap", type=int, default=BLOCK_OVERLAP)
    parser.add_argument("--context", type=int, default=BLOCK_CONTEXT)
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute patches that already have a cell_mask.npy")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.slide:
        slide = cv2.imread(args.slide, cv2.IMREAD_COLOR)
        if slide is None:
            raise FileNotFoundError(args.slide)
        rgb = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)

        labels, polys = segment_section(model, rgb, args.block_size,
                                        args.min_overlap, args.context)
        labels = drop_small_objects(labels, args.min_area)
        cells = expand_non_overlapping(labels, args.distance_um, args.mpp)

        out_path = Path(args.out) if args.out else Path(args.slide).with_suffix(".cell_labels.npy")
        np.save(out_path, cells)
        print(f"wrote {out_path}  ({len(np.unique(cells)) - 1} cells)")

        if args.polys:
            with open(args.polys, "wb") as handle:
                pickle.dump(polys, handle)
            print(f"wrote {args.polys}")
        return

    if not args.root:
        parser.error("--root is required with --tile / --all")
    root = Path(args.root)
    if args.tile:
        targets = [root / args.tile]
    elif args.all:
        targets = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        parser.error("give --tile, --all or --slide")

    total = 0
    for patch_dir in targets:
        status, count = process_tile(model, patch_dir, args.distance_um,
                                     args.min_area, overwrite=args.overwrite)
        total += count
        print(f"[{patch_dir.name}] {status}" + (f" ({count} cells)" if count else ""))
    print(f"done, {total} cells across {len(targets)} patch(es)")


if __name__ == "__main__":
    main()
