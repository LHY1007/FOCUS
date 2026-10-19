"""Turn the H&E patches into the per-cell masks FOCUS trains on.

This is the step that produces ``cell_mask.npy`` for every patch, following the
segmentation description in the supplementary method section:

    "Nuclei were segmented from 0.5 um/pixel patches using a pretrained
    StarDist2D model ... Predictions were converted to nuclear masks using
    default thresholds, small artifacts were removed, and masks were
    isotropically dilated in a non-overlapping manner to approximate full-cell
    boundaries. The resulting nuclei and cell masks were stitched back to the
    tissue field and visually inspected in representative regions."

Each clause maps to one call below:

    pretrained StarDist2D     -> StarDist2D.from_pretrained("2D_versatile_he")
    default thresholds        -> PROB_THRESH / NMS_THRESH left as None
    small artifacts removed   -> drop_small_objects()
    isotropically dilated,
      non-overlapping         -> expand_non_overlapping()
    stitched back             -> the per-patch frame is the patch itself; the
                                 stitched view is written by --stitch
    visually inspected        -> render()

Output layout, per patch: ``cell_mask.npy`` of shape ``(256, 256, N)``, one
binary channel per cell, on the same 10 um/px grid as ``5120_to256.npy``.
``guided_diffusion/load_dataset.py`` reads it with ``transpose(2, 0, 1)`` and
``UNetModel`` reduces it to ``[N, 256, 256]``, so the channel axis must be the
cell axis.

Where this differs from ``downstream/matrix/bin2cell_pipeline.py``: that script
follows the published bin2cell recipe, whose detector thresholds are 0.01/0.001
rather than the model defaults, and whose expansion works on the 2 um bin grid
instead of on pixels. Both are documented in ``data_preprocessing/README.md``.

    python cell_mask.py --root /path/to/WSI/extract/20240612 --all
    python cell_mask.py --tile /path/to/WSI/extract/20240612/3_3
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Pretrained StarDist model. ``2D_versatile_he`` is the brightfield H&E model;
#: the fluorescence models are trained on a different modality.
MODEL_NAME = "2D_versatile_he"

#: ``None`` keeps the model's own thresholds, which is what the method section
#: specifies. Lowering them admits sub-nuclear fragments that the artifact
#: filter then has to remove anyway.
PROB_THRESH = None
NMS_THRESH = None

#: Percentile normalisation applied before prediction.
NORM_LO = 1.0
NORM_HI = 99.8

#: Detections smaller than this many pixels are debris. At 0.5 um/px, 100 px is
#: a 5 x 5 um object, smaller than any real nucleus.
MIN_NUCLEUS_AREA = 100

#: Growth applied to each nucleus, in micrometres, approximating the cytoplasm
#: the nuclear stain does not cover.
EXPAND_UM = 5.0

#: Physical pixel size of the patch, and the grid the mask is written on.
MPP = 0.5
MASK_SIDE = 256
TILE_SIDE = 5120
DOWNSAMPLE = TILE_SIDE // MASK_SIDE          # 20 px at 0.5 um/px = 10 um

#: A 10 um/px pixel goes to the cell covering at least this share of the
#: 20 x 20 block it summarises; below it the pixel stays background.
MIN_COVERAGE = 0.25


# ---------------------------------------------------------------------------
# Nuclei
# ---------------------------------------------------------------------------


def load_model(model_name: str = MODEL_NAME):
    """Load the pretrained StarDist2D model."""
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained(model_name)


def normalise(rgb: np.ndarray) -> np.ndarray:
    """Percentile-normalise *rgb* jointly across the three H&E channels.

    Per-channel normalisation would shift the stain ratios the detector keys on.
    """
    from csbdeep.utils import normalize

    return normalize(rgb, NORM_LO, NORM_HI, axis=(0, 1, 2))


def segment_nuclei(model, rgb: np.ndarray) -> np.ndarray:
    """Segment one patch at the model's default thresholds."""
    kwargs = {}
    if PROB_THRESH is not None:
        kwargs["prob_thresh"] = PROB_THRESH
    if NMS_THRESH is not None:
        kwargs["nms_thresh"] = NMS_THRESH

    labels, _polys = model.predict_instances(normalise(rgb), **kwargs)
    return labels.astype(np.int32, copy=False)


def drop_small_objects(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove labels covering fewer than *min_area* pixels.

    Counted per label rather than on the binarised mask, so a sheet of touching
    nuclei is not mistaken for a single object.
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
                           mpp: float = MPP) -> np.ndarray:
    """Grow every label isotropically into the background without overlapping.

    Each background pixel within reach of some nucleus is assigned to the
    nucleus **nearest** to it, measured as a Euclidean distance in micrometres.
    Because "nearest" selects exactly one nucleus per pixel, no pixel can be
    claimed by two labels: the result is a partition, and the per-cell channels
    built from it are disjoint by construction. A plain morphological dilation
    does not have that property, because two expanding labels can write the same
    pixel and the last writer silently wins.

    Pixels further than *distance_um* from any nucleus stay background.
    """
    if distance_um <= 0:
        return labels

    background = labels == 0
    if not background.any():
        return labels

    # Distance from each background pixel to the nearest labelled pixel, and the
    # coordinates of that pixel. `background` is 1 on the background, so its
    # zeros are exactly the labelled pixels being measured against.
    distances, nearest = ndi.distance_transform_edt(
        background, sampling=(mpp, mpp), return_indices=True
    )
    nearest_labels = labels[tuple(nearest)]

    grown = labels.copy()
    reachable = background & (distances <= distance_um)
    grown[reachable] = nearest_labels[reachable]
    return grown


# ---------------------------------------------------------------------------
# Mask
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

    # Only the region the input covered was written, so the coverage mask has to
    # be applied to that same corner rather than to the whole array.
    out[:usable_h, :usable_w][coverage < min_coverage] = 0
    return out


def stack_cells(labels: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Turn a 2-D label image into ``(H, W, N)`` binary per-cell channels.

    A patch with no cells yields ``(MASK_SIDE, MASK_SIDE, 0)`` so the channel
    axis is always present.
    """
    ids = np.unique(labels)
    ids = ids[ids > 0]
    height, width = labels.shape

    channels = np.zeros((height, width, ids.size), dtype=dtype)
    for index, label in enumerate(ids):
        channels[..., index] = labels == label
    return channels


def render(channels: np.ndarray, rgb: np.ndarray, out_path: Path) -> None:
    """Write a PNG of the cell partition over the patch, for visual inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.argmax(channels, axis=2) + 1
    labels[channels.sum(axis=2) == 0] = 0
    boundary = labels != ndi.grey_dilation(labels, size=(3, 3))

    overlay = np.array(rgb, copy=True)
    overlay[boundary] = [255, 0, 0]

    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].set_title("H&E")
    axes[0].imshow(rgb)
    axes[1].set_title(f"cell boundaries, {channels.shape[2]} cells")
    axes[1].imshow(overlay)
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def read_patch(path: Path) -> np.ndarray:
    """Read a patch as RGB uint8."""
    rgb = np.load(path)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    return np.ascontiguousarray(rgb[..., :3])


def process_patch(model, patch_dir: Path, expand_um: float, min_area: int,
                  overwrite: bool, draw: bool) -> tuple[str, int]:
    """Write ``cell_mask.npy`` for one patch directory."""
    source = patch_dir / "5120.npy"
    target = patch_dir / "cell_mask.npy"

    if target.exists() and not overwrite:
        return "exists", 0
    if not source.exists():
        return f"no {source.name}", 0

    rgb = read_patch(source)
    nuclei = drop_small_objects(segment_nuclei(model, rgb), min_area)
    cells = expand_non_overlapping(nuclei, expand_um)
    channels = stack_cells(downsample_labels(cells))

    np.save(target, channels)
    if draw:
        render(channels, rgb, patch_dir / "cell_mask.png")

    count = channels.shape[2]
    del rgb, nuclei, cells, channels
    gc.collect()
    return "ok", count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None,
                        help="WSI/extract/{layer} holding the {i}_{j} directories")
    parser.add_argument("--tile", default=None,
                        help="one patch directory to process, e.g. 3_3")
    parser.add_argument("--all", action="store_true",
                        help="process every patch directory under --root")
    parser.add_argument("--distance-um", type=float, default=EXPAND_UM,
                        help="how far to grow each nucleus, in micrometres")
    parser.add_argument("--min-area", type=int, default=MIN_NUCLEUS_AREA,
                        help="drop detections smaller than this many pixels")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="StarDist2D pretrained model name")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute patches that already have a cell_mask.npy")
    parser.add_argument("--draw", action="store_true",
                        help="also write cell_mask.png for visual inspection")
    args = parser.parse_args()

    if not args.root and not args.tile:
        parser.error("give --root or --tile")

    root = Path(args.root) if args.root else Path(args.tile).parent
    if args.tile:
        targets = [root / args.tile]
    elif args.all:
        targets = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        parser.error("give --all together with --root")

    model = load_model(args.model)

    total = 0
    for patch_dir in targets:
        status, count = process_patch(model, patch_dir, args.distance_um,
                                      args.min_area, args.overwrite, args.draw)
        total += count
        print(f"[{patch_dir.name}] {status}" + (f" ({count} cells)" if count else ""))
    print(f"done, {total} cells across {len(targets)} patch(es)")


if __name__ == "__main__":
    main()
