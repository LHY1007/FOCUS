"""Nuclei and cell segmentation of the H&E.

The recipe, in the order the steps are applied:

  * the H&E is cropped to the tissue-covered area with an intensity threshold
    followed by a morphological closing;
  * nuclei are segmented from the 0.5 um/px patches with a pretrained StarDist2D
    model running at its **default thresholds**;
  * small detections are removed;
  * the nuclei are grown **isotropically into a non-overlapping** partition, so
    each label approximates a full cell boundary rather than only the nucleus;
  * the labels are stitched back over the tissue field and rendered for visual
    inspection.

The output is the segmentation map: one integer label per cell, 0 for
background.

    python stardist_segmentation.py --slide section.tif --out labels.npy
    python stardist_segmentation.py --tile /path/to/WSI/extract/20240612/3_3
    python stardist_segmentation.py --root /path/to/WSI/extract/20240612 --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Pretrained StarDist model. ``2D_versatile_he`` is the brightfield H&E model;
#: the fluorescence models are trained on a different modality and are not
#: interchangeable with it.
MODEL_NAME = "2D_versatile_he"

#: ``None`` keeps the model's own thresholds. They are the values to use here:
#: lowering them admits sub-nuclear fragments that the artifact filter below then
#: has to remove anyway.
PROB_THRESH = None
NMS_THRESH = None

#: Percentile normalisation applied before prediction.
NORM_LO = 1.0
NORM_HI = 99.8

#: Block geometry for whole-slide prediction. Objects must be smaller than
#: ``BLOCK_OVERLAP``, and ``BLOCK_OVERLAP + 2 * BLOCK_CONTEXT < BLOCK_SIZE``.
BLOCK_SIZE = 4096
BLOCK_OVERLAP = 128
BLOCK_CONTEXT = 128

#: Detections smaller than this many pixels are debris. At 0.5 um/px, 100 px is
#: a 5 x 5 um object, smaller than any real nucleus.
MIN_NUCLEUS_AREA = 100

#: Growth applied to each nucleus, in micrometres -- the cytoplasm the nuclear
#: stain does not cover.
EXPAND_UM = 5.0

#: Physical pixel size of the patches.
MPP = 0.5

#: Tissue mask: Otsu threshold, then closing to fill holes inside tissue.
CLOSE_RADIUS = 10

#: A patch below this share of tissue is background and is skipped.
MIN_TISSUE_COVERAGE = 0.05

#: Tile side of the patch layout, used when stitching patch directories back.
TILE_SIDE = 5120


# ---------------------------------------------------------------------------
# Tissue mask
# ---------------------------------------------------------------------------


def otsu_threshold(gray: np.ndarray, bins: int = 256) -> float:
    """Return Otsu's threshold for a uint8 grayscale image.

    The between-class variance is evaluated at every histogram bin and its
    maximum taken. Background glass is bright and near-uniform, so this separates
    it from stained tissue without a fixed intensity that would need tuning per
    slide.
    """
    hist, edges = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0

    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_total = float(np.dot(hist, centres))
    sum_bg = np.cumsum(hist * centres)

    # A threshold that puts every pixel on one side has no variance to maximise,
    # so those bins are excluded rather than divided by zero.
    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.divide(sum_bg, weight_bg, out=np.zeros_like(sum_bg),
                        where=weight_bg > 0)
    mean_fg = np.divide(sum_total - sum_bg, weight_fg,
                        out=np.zeros_like(sum_bg), where=weight_fg > 0)
    variance = np.zeros_like(centres)
    variance[valid] = (weight_bg[valid] * weight_fg[valid]
                       * (mean_bg[valid] - mean_fg[valid]) ** 2)
    return float(centres[int(np.argmax(variance))])


def disk(radius: int) -> np.ndarray:
    """Return a circular structuring element of the given radius."""
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ((xx ** 2 + yy ** 2) <= radius ** 2).astype(bool)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luminance of *rgb* as uint8."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = rgb.astype(np.float32) @ weights
    return np.clip(gray, 0, 255).astype(np.uint8)


def tissue_mask(rgb: np.ndarray, close_radius: int = CLOSE_RADIUS) -> np.ndarray:
    """Boolean mask of the tissue-covered area of *rgb*.

    Grayscale, Otsu threshold with the tissue as foreground, then a morphological
    closing so holes and cracks inside the tissue are filled rather than carved
    out of the crop.
    """
    foreground = to_gray(rgb) <= otsu_threshold(to_gray(rgb))
    return ndi.binary_closing(foreground, structure=disk(close_radius))


# ---------------------------------------------------------------------------
# Nuclei
# ---------------------------------------------------------------------------


def load_model(model_name: str = MODEL_NAME):
    """Load the pretrained StarDist2D model."""
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained(model_name)


def normalise(rgb: np.ndarray) -> np.ndarray:
    """Percentile-normalise *rgb* jointly across the three H&E channels.

    Per-channel normalisation would shift the stain ratios the detector keys on,
    so the channels are scaled together.
    """
    from csbdeep.utils import normalize

    return normalize(rgb, NORM_LO, NORM_HI, axis=(0, 1, 2))


def _thresholds() -> dict:
    kwargs = {}
    if PROB_THRESH is not None:
        kwargs["prob_thresh"] = PROB_THRESH
    if NMS_THRESH is not None:
        kwargs["nms_thresh"] = NMS_THRESH
    return kwargs


def segment_patch(model, rgb: np.ndarray) -> np.ndarray:
    """Segment a patch that fits in memory; return an int32 label image."""
    labels, _polys = model.predict_instances(normalise(rgb), **_thresholds())
    return labels.astype(np.int32, copy=False)


def segment_slide(model, rgb: np.ndarray, block_size: int = BLOCK_SIZE,
                  min_overlap: int = BLOCK_OVERLAP,
                  context: int = BLOCK_CONTEXT) -> np.ndarray:
    """Segment a whole section in blocks and stitch the labels back together.

    Each block is predicted on its own, its context ring is discarded and the
    surviving objects are relabelled, so the stitched result matches a
    single-pass prediction while the whole slide is never held in memory. The
    assumption is that no object is larger than *min_overlap*.
    """
    labels, _polys = model.predict_instances_big(
        normalise(rgb),
        axes="YXC",
        block_size=block_size,
        min_overlap=min_overlap,
        context=context,
        normalizer=None,
        **_thresholds(),
    )
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
    claimed by two labels: the result is a partition, and any per-cell weighting
    built from it is disjoint by construction. A plain morphological dilation
    does not have that property, because two expanding labels can write the same
    pixel and the last writer silently wins.

    Pixels further than *distance_um* from any nucleus stay background. Ties are
    resolved by the distance transform's own ordering, which is deterministic.

    Parameters
    ----------
    labels:
        Integer label image, 0 for background.
    distance_um:
        How far to grow, in micrometres. ``0`` returns *labels* unchanged.
    mpp:
        Micrometres per pixel of *labels*, so the growth is isotropic in physical
        units rather than in pixels.
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


def stitch(blocks: list[tuple[tuple[int, int], np.ndarray]],
           tile_side: int = TILE_SIDE) -> np.ndarray:
    """Place labelled patches back on the tissue field in one canvas.

    Patch folders are named ``{i}_{j}``; ``i`` indexes columns of the layout and
    ``j`` rows, so the block is placed at ``(j * tile_side, i * tile_side)``.
    Labels are shifted by a running total so that two patches can never collide
    on the canvas.
    """
    height = max(y + block.shape[0] for (y, _x), block in blocks)
    width = max(x + block.shape[1] for (_y, x), block in blocks)
    canvas = np.zeros((height, width), dtype=np.int32)

    offset = 0
    for (y, x), block in blocks:
        shifted = block.astype(np.int32, copy=True)
        shifted[shifted > 0] += offset
        canvas[y:y + block.shape[0], x:x + block.shape[1]] = shifted
        offset = int(shifted.max())
    return canvas


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def render(labels: np.ndarray, rgb: np.ndarray | None, out_path: Path) -> None:
    """Write a PNG of the segmentation map for visual inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 3 if rgb is not None else 1
    figure, grid = plt.subplots(1, panels, figsize=(7 * panels, 7))
    grid = np.atleast_1d(grid)

    if rgb is not None:
        grid[0].set_title("H&E")
        grid[0].imshow(rgb)
        grid[0].set_axis_off()

    grid[-1 if rgb is None else 1].set_title(
        f"segmentation, {int(labels.max())} cells")
    grid[-1 if rgb is None else 1].imshow(labels, cmap="nipy_spectral")
    grid[-1 if rgb is None else 1].set_axis_off()

    if rgb is not None:
        # Boundaries over the image: a mis-segmentation shows up as a boundary
        # nowhere near a nucleus, which a colour map alone would hide.
        boundary = labels != ndi.grey_dilation(labels, size=(3, 3))
        overlay = np.array(rgb, copy=True)
        overlay[boundary] = [255, 0, 0]
        grid[2].set_title("boundaries over H&E")
        grid[2].imshow(overlay)
        grid[2].set_axis_off()

    for axis in grid:
        axis.set_axis_off()

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def read_image(path: Path) -> np.ndarray:
    """Read an image as RGB uint8, whichever library can open it."""
    try:
        import tifffile
        array = tifffile.imread(str(path))
    except Exception:                             # noqa: BLE001
        import matplotlib.image as mpimg
        array = mpimg.imread(str(path))

    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.dtype != np.uint8:
        peak = float(array.max()) or 1.0
        array = (array / peak * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[..., :3])


def process(model, rgb: np.ndarray, distance_um: float, min_area: int,
            whole_slide: bool) -> np.ndarray:
    """Segment one image and return the expanded label image."""
    raw = segment_slide(model, rgb) if whole_slide else segment_patch(model, rgb)
    raw = drop_small_objects(raw, min_area)
    print(f"  {int(raw.max())} nuclei")
    cells = expand_non_overlapping(raw, distance_um, MPP)
    print(f"  {int(cells.max())} cells after expansion")
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slide", default=None,
                        help="one image to segment in a single pass")
    parser.add_argument("--tile", default=None,
                        help="a patch directory holding 5120.npy")
    parser.add_argument("--root", default=None,
                        help="patch root holding the {i}_{j} directories")
    parser.add_argument("--all", action="store_true",
                        help="process every patch under --root and stitch them")
    parser.add_argument("--out", default=None,
                        help="output path for the label image (--slide)")
    parser.add_argument("--distance-um", type=float, default=EXPAND_UM,
                        help="how far to grow each nucleus, in micrometres")
    parser.add_argument("--min-area", type=int, default=MIN_NUCLEUS_AREA,
                        help="drop detections smaller than this many pixels")
    parser.add_argument("--no-expand", action="store_true",
                        help="keep the nuclear masks instead of the cell masks")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="StarDist2D pretrained model name")
    args = parser.parse_args()

    model = load_model(args.model)
    distance = 0.0 if args.no_expand else args.distance_um

    if args.slide:
        path = Path(args.slide)
        rgb = read_image(path)
        print(f"{path.name}: {rgb.shape[1]} x {rgb.shape[0]} px")
        print(f"  tissue covers {tissue_mask(rgb).mean():.1%} of the frame")

        labels = process(model, rgb, distance, args.min_area, whole_slide=True)
        out_path = Path(args.out) if args.out else path.with_suffix(".labels.npy")
        np.save(out_path, labels)
        print(f"  wrote {out_path}")

        render_path = out_path.with_suffix(".png")
        render(labels, rgb, render_path)
        print(f"  wrote {render_path}")
        return

    root = args.root or (str(Path(args.tile).parent) if args.tile else None)
    if not root:
        parser.error("give --slide, --tile or --root")

    root = Path(root)
    if args.tile:
        targets = [root / args.tile]
    elif args.all:
        targets = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        parser.error("give --tile, --all or --slide")

    blocks = []
    for patch_dir in targets:
        source = patch_dir / "5120.npy"
        if not source.exists():
            print(f"[{patch_dir.name}] no 5120.npy")
            continue

        rgb = read_image(source)
        if tissue_mask(rgb).mean() < MIN_TISSUE_COVERAGE:
            print(f"[{patch_dir.name}] background")
            continue

        labels = process(model, rgb, distance, args.min_area, whole_slide=False)
        np.save(patch_dir / "labels.npy", labels)
        blocks.append((tuple(int(v) for v in patch_dir.name.split("_")), labels))
        print(f"[{patch_dir.name}] wrote labels.npy")

    if len(blocks) > 1:
        placed = [((j * TILE_SIDE, i * TILE_SIDE), block)
                  for (i, j), block in blocks]
        canvas = stitch(placed)
        out_path = root / "labels_stitched.npy"
        np.save(out_path, canvas)
        print(f"stitched {len(blocks)} patches -> {out_path}")
        render(canvas, None, root / "labels_stitched.png")


if __name__ == "__main__":
    main()
