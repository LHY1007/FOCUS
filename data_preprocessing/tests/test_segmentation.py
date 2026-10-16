"""Checks for the cell-segmentation helpers.

Runs without a GPU, a StarDist model or a real slide: the detector is replaced by
synthetic label images, so what is tested is the geometry -- which is where the
non-overlap guarantee and the channel layout live, and the two places a change
would silently corrupt ``cell_mask.npy``.

    python test_segmentation.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "histology"))

# The geometry checks below -- the expansion, the channel layout and the block
# reduction -- need only numpy and scipy, and those are the parts worth running
# everywhere. OpenCV is used by the tissue mask alone, so when it is absent it is
# stubbed out and that section is reported as skipped rather than failing.
try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    _stub = types.ModuleType("cv2")
    _stub.COLOR_RGB2GRAY = 4
    _stub.THRESH_BINARY_INV = 1
    _stub.THRESH_OTSU = 8
    _stub.MORPH_CLOSE = 3
    _stub.MORPH_OPEN = 2

    def _unavailable(*args, **kwargs):
        raise RuntimeError("opencv is not installed")

    _stub.cvtColor = _stub.threshold = _stub.morphologyEx = _unavailable
    sys.modules["cv2"] = _stub

from scipy import ndimage as ndi  # noqa: E402

from stardist_segmentation import (  # noqa: E402
    MASK_SIDE,
    disk,
    downsample_labels,
    drop_small_objects,
    expand_non_overlapping,
    stack_cells,
    tissue_mask,
)

failures = []
skipped = []


def check(label: str, condition: bool, detail="") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        failures.append(label)
        print(f"  FAIL  {label} {detail}")


def skip(label: str, why: str) -> None:
    skipped.append(label)
    print(f"  SKIP  {label} ({why})")


def synthetic_labels(shape=(200, 200), gap=4):
    """Two large nuclei *gap* pixels apart, plus one small speck of debris."""
    labels = np.zeros(shape, dtype=np.int32)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    centre = shape[1] // 2
    for value, offset, radius in ((1, -(gap // 2 + 12), 12), (2, gap // 2 + 12, 12)):
        mask = ((yy - 50) ** 2 + (xx - (centre + offset)) ** 2) <= radius ** 2
        labels[mask] = value
    speck = ((yy - 150) ** 2 + (xx - 150) ** 2) <= 2 ** 2
    labels[speck] = 3
    return labels


print("== disk ==")
d = disk(3)
check("square and odd-sized", d.shape == (7, 7), d.shape)
check("centre set", d[3, 3] == 1)
check("corner clear", d[0, 0] == 0)
check("symmetric", np.array_equal(d, d[::-1]) and np.array_equal(d, d[:, ::-1]))

print("== drop_small_objects ==")
labels = synthetic_labels()
kept = drop_small_objects(labels, min_area=100)
check("large nuclei kept", set(np.unique(kept)) == {0, 1, 2}, np.unique(kept))
check("speck removed", 3 not in np.unique(kept))

print("== expand_non_overlapping: bounding ==")
nuclei = drop_small_objects(synthetic_labels(), min_area=100)
grown = expand_non_overlapping(nuclei, distance_um=5.0, mpp=0.5)   # 10 px
check("labels preserved", set(np.unique(grown)) == {0, 1, 2}, np.unique(grown))
check("region grew", (grown > 0).sum() > (nuclei > 0).sum(),
      f"{(nuclei > 0).sum()} -> {(grown > 0).sum()}")

reach = 10  # px
far = np.zeros_like(nuclei)
far[0, 0] = 9                                     # an isolated label in the corner
spread = expand_non_overlapping(far, distance_um=reach * 0.5, mpp=0.5)
rows, cols = np.where(spread > 0)
check("growth respects the distance", rows.max() <= reach and cols.max() <= reach,
      (rows.max(), cols.max()))
check("nothing beyond the distance", spread[reach + 1:, reach + 1:].sum() == 0)

print("== expand_non_overlapping: isotropy ==")
# A single nucleus must grow into a disk: the reach has to be the same in every
# direction for the expansion to be isotropic.
single = np.zeros((101, 101), dtype=np.int32)
single[50, 50] = 1
spread = expand_non_overlapping(single, distance_um=10.0, mpp=0.5)   # 20 px
extent = {
    "left": 50 - np.where(spread[50] > 0)[0].min(),
    "right": np.where(spread[50] > 0)[0].max() - 50,
    "up": 50 - np.where(spread[:, 50] > 0)[0].min(),
    "down": np.where(spread[:, 50] > 0)[0].max() - 50,
}
check("isotropic", len(set(extent.values())) == 1, extent)
check("diagonal is shorter than the axis", spread[50 - 20, 50 - 20] == 0)

print("== expand_non_overlapping: the non-overlap guarantee ==")
# The whole point of using nearest-nucleus assignment: two nuclei close enough
# that their growth regions meet must still partition the pixels between them.
nuclei = drop_small_objects(synthetic_labels(gap=4), min_area=100)
grown = expand_non_overlapping(nuclei, distance_um=20.0, mpp=0.5)  # 40 px, well past meeting
channels = stack_cells(grown)
check("no pixel belongs to two cells", channels.sum(axis=2).max() <= 1,
      channels.sum(axis=2).max())
check("the two cells meet", grown[50, 95:106].min() > 0,
      grown[50, 95:106].tolist())

# The same expansion done per label with a morphological dilation double-writes
# the contested pixels; this is what the nearest-nucleus rule avoids, and the
# test is here so that a future rewrite to a plain dilation fails loudly.
naive = np.zeros_like(nuclei)
for value in (1, 2):
    naive += ndi.binary_dilation(nuclei == value,
                                 structure=disk(40)).astype(np.int32)
check("naive dilation does overlap, which is why it is not used",
      naive.max() > 1, naive.max())

print("== stack_cells ==")
channels = stack_cells(downsample_labels(grown))
check("shape is (H, W, N)", channels.shape[:2] == (MASK_SIDE, MASK_SIDE), channels.shape)
check("one channel per cell", channels.shape[2] == 2, channels.shape)
check("binary", set(np.unique(channels)) <= {0, 1}, np.unique(channels))
check("channels disjoint", channels.sum(axis=2).max() <= 1, channels.sum(axis=2).max())

empty = stack_cells(np.zeros((MASK_SIDE, MASK_SIDE), dtype=np.int32))
check("empty patch keeps the channel axis",
      empty.shape == (MASK_SIDE, MASK_SIDE, 0), empty.shape)

print("== downsample_labels ==")
solid = np.zeros((40, 40), dtype=np.int32)
solid[:20, :20] = 7
coarse = downsample_labels(solid, out_side=2, factor=20, min_coverage=0.5)
check("majority label wins the block", coarse[0, 0] == 7, coarse.tolist())
check("background blocks stay zero", coarse[1, 1] == 0, coarse.tolist())

clipping = np.zeros((40, 40), dtype=np.int32)
clipping[:4, :4] = 5                              # 4 % of the block
coarse = downsample_labels(clipping, out_side=2, factor=20, min_coverage=0.5)
check("a clipping label is not promoted", coarse[0, 0] == 0, coarse.tolist())

print("== tissue_mask ==")
if not HAVE_CV2:
    skip("tissue_mask", "opencv not installed")
else:
    glass = np.full((200, 200, 3), 240, dtype=np.uint8)
    check("uniform glass has no tissue", tissue_mask(glass, 3, 2).mean() < 0.01,
          tissue_mask(glass, 3, 2).mean())

    stained = glass.copy()
    stained[60:140, 60:140] = (120, 80, 150)      # 80 x 80 of 200 x 200 = 16 %
    mask = tissue_mask(stained, 3, 2)
    check("stained block detected", mask[100, 100])
    check("glass still background", not mask[10, 10])
    check("mask covers the block, not the glass", 0.12 < mask.mean() < 0.20,
          f"{mask.mean():.4f}")

print()
if skipped:
    print(f"{len(skipped)} section(s) skipped: {', '.join(skipped)}")
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("segmentation helpers behave as expected")
