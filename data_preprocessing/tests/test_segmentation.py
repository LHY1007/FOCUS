"""Checks for the segmentation helpers.

Runs without a GPU, a StarDist model or a real slide: the detector is replaced by
synthetic label images, so what is tested is the geometry -- which is where the
non-overlap guarantee lives, and the place a change would silently corrupt the
segmentation map. Only numpy and scipy are needed.

    python test_segmentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "histology"))

from stardist_segmentation import (  # noqa: E402
    disk,
    drop_small_objects,
    expand_non_overlapping,
    otsu_threshold,
    stitch,
    tissue_mask,
)

failures = []


def check(label: str, condition: bool, detail="") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        failures.append(label)
        print(f"  FAIL  {label} {detail}")


def synthetic_labels(shape=(200, 200), gap=4):
    """Two large nuclei *gap* pixels apart, plus one small speck of debris."""
    labels = np.zeros(shape, dtype=np.int32)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    centre = shape[1] // 2
    for value, offset, radius in ((1, -(gap // 2 + 12), 12), (2, gap // 2 + 12, 12)):
        labels[((yy - 50) ** 2 + (xx - (centre + offset)) ** 2) <= radius ** 2] = value
    labels[((yy - 150) ** 2 + (xx - 150) ** 2) <= 2 ** 2] = 3
    return labels


print("== disk ==")
d = disk(3)
check("square and odd-sized", d.shape == (7, 7), d.shape)
check("centre set", bool(d[3, 3]))
check("corner clear", not d[0, 0])
check("symmetric", np.array_equal(d, d[::-1]) and np.array_equal(d, d[:, ::-1]))

print("== otsu_threshold ==")
# A bimodal image: 40 % dark tissue over a bright background. Otsu has to land
# between the two modes, not on one of them.
bimodal = np.concatenate([np.full(4000, 40, np.uint8),
                          np.full(6000, 230, np.uint8)])
threshold = otsu_threshold(bimodal)
check("separates the two modes", 40 < threshold < 230, threshold)
# Degenerate inputs must not divide by zero: with nothing to separate, any
# threshold in range is acceptable, it just has to come back without raising.
check("empty image does not raise",
      0.0 <= otsu_threshold(np.zeros(10, np.uint8)) <= 256.0)
check("single-valued image does not raise",
      0.0 <= otsu_threshold(np.full(10, 7, np.uint8)) <= 256.0)

print("== tissue_mask ==")
glass = np.full((200, 200, 3), 240, dtype=np.uint8)
check("uniform glass has no tissue", tissue_mask(glass, 3).mean() < 0.01,
      tissue_mask(glass, 3).mean())

stained = glass.copy()
stained[60:140, 60:140] = (120, 80, 150)          # 80 x 80 of 200 x 200 = 16 %
mask = tissue_mask(stained, 3)
check("stained block detected", bool(mask[100, 100]))
check("glass still background", not mask[10, 10])
check("mask covers the block, not the glass", 0.12 < mask.mean() < 0.20,
      f"{mask.mean():.4f}")

# Closing must fill a hole inside the tissue rather than leave a crater in it.
# The hole is 10 px across, so it takes the default radius of 10 to close.
holed = stained.copy()
holed[95:105, 95:105] = (240, 240, 240)
check("closing fills holes in tissue", bool(tissue_mask(holed)[100, 100]))
check("a small radius does not close it",
      not tissue_mask(holed, 2)[100, 100])

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

reach = 10
corner = np.zeros_like(nuclei)
corner[0, 0] = 9
spread = expand_non_overlapping(corner, distance_um=reach * 0.5, mpp=0.5)
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
check("diagonal is shorter than the axis", spread[30, 30] == 0)

print("== expand_non_overlapping: the non-overlap guarantee ==")
# The point of nearest-nucleus assignment: two nuclei close enough that their
# growth regions meet must still partition the pixels between them.
nuclei = drop_small_objects(synthetic_labels(gap=4), min_area=100)
grown = expand_non_overlapping(nuclei, distance_um=20.0, mpp=0.5)
per_label = np.stack([grown == value for value in (1, 2)])
check("no pixel belongs to two cells", per_label.sum(axis=0).max() <= 1,
      per_label.sum(axis=0).max())
check("the two cells meet", grown[50, 95:106].min() > 0,
      grown[50, 95:106].tolist())

# The same expansion done per label with a morphological dilation double-writes
# the contested pixels. This is what the nearest-nucleus rule avoids, and the
# check is here so a future rewrite to a plain dilation fails loudly.
naive = np.zeros_like(nuclei)
for value in (1, 2):
    naive += ndi.binary_dilation(nuclei == value, structure=disk(40)).astype(np.int32)
check("naive dilation does overlap, which is why it is not used",
      naive.max() > 1, naive.max())

print("== stitch ==")
a = np.array([[1, 1], [0, 2]], dtype=np.int32)
b = np.array([[1, 0], [3, 3]], dtype=np.int32)
canvas = stitch([((0, 0), a), ((0, 2), b)])
check("canvas shape", canvas.shape == (2, 4), canvas.shape)
check("left block in place", canvas[0, 0] == 1 and canvas[1, 1] == 2,
      canvas.tolist())
check("second block shifted past the first, so the two cannot collide",
      set(np.unique(canvas)) == {0, 1, 2, 3, 5}, np.unique(canvas))
check("no label lost", (canvas > 0).sum() == (a > 0).sum() + (b > 0).sum(),
      (canvas > 0).sum())

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("segmentation helpers behave as expected")
