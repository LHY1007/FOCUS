"""Cell-mask measurements for Methods 4; this module does not generate masks."""
import math
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2hed
from skimage.measure import regionprops


@torch.no_grad()
def complexity_grid(he, masks, patch_size=16, pixel_size_um=10.0):
    """Return [B,H/p,W/p] scores using cell density, shape, size and texture.

    Descriptor deviations are mean absolute z scores relative to all cells in
    the same input tile. This normalization is a new, explicit implementation
    choice because Methods 4 does not specify the reference for its z values.
    Cells are assigned to one patch by centroid; empty patches score zero.
    """
    results = []
    for image, stack in zip(he.detach().cpu().numpy(), masks.detach().cpu().numpy()):
        rgb = np.moveaxis(image[:3], 0, -1).astype(float)
        if rgb.max() > 1:
            rgb /= 255.0
        hematoxylin = rgb2hed(np.clip(rgb, 0, 1))[..., 0]
        h, w = rgb.shape[:2]
        if h % patch_size or w % patch_size:
            raise ValueError('H&E dimensions must be divisible by cellular patch size')
        rows, locations = [], []
        for mask in stack:
            binary = mask > 0
            if not binary.any():
                continue
            prop = regionprops(binary.astype(np.uint8))[0]
            values = hematoxylin[binary]
            hist, _ = np.histogram(values, bins=32, range=(float(hematoxylin.min()),
                                                          float(hematoxylin.max()) + 1e-8))
            prob = hist[hist > 0] / max(hist.sum(), 1)
            entropy = -(prob * np.log2(prob)).sum()
            rows.append([prop.eccentricity, prop.solidity,
                         min(4 * math.pi * prop.area / max(prop.perimeter ** 2, 1), 1),
                         math.log(max(prop.area * pixel_size_um ** 2, 1e-8)),
                         prop.equivalent_diameter_area * pixel_size_um,
                         entropy, values.var()])
            locations.append((min(int(prop.centroid[0]) // patch_size, h // patch_size - 1),
                              min(int(prop.centroid[1]) // patch_size, w // patch_size - 1)))
        score = np.zeros((h // patch_size, w // patch_size), dtype=np.float32)
        if rows:
            desc = np.asarray(rows)
            std = desc.std(0)
            z = np.abs((desc - desc.mean(0)) / np.where(std > 1e-8, std, 1.0))
            nmax = (patch_size * pixel_size_um) ** 2 / (math.pi * 3 ** 2)
            locations = np.asarray(locations)
            for r in range(score.shape[0]):
                for c in range(score.shape[1]):
                    idx = (locations[:, 0] == r) & (locations[:, 1] == c)
                    if not idx.any():
                        continue
                    delta = np.array([z[idx, :3].mean(), z[idx, 3:5].mean(), z[idx, 5:].mean()])
                    heterogeneity = (delta / (1 + delta)).mean()
                    score[r, c] = 0.5 * min(idx.sum() / nmax, 1) + 0.5 * heterogeneity
        results.append(score)
    return torch.as_tensor(np.stack(results), device=he.device, dtype=he.dtype)


@torch.no_grad()
def boundary_weights(masks, size):
    """Normalized gradient of cell-mask distance transforms, Methods 1.4."""
    maps = []
    for stack in masks.detach().cpu().numpy():
        distance = sum((distance_transform_edt(m > 0) for m in stack),
                       np.zeros(stack.shape[-2:], dtype=np.float32))
        gy, gx = np.gradient(distance)
        grad = np.hypot(gx, gy)
        maps.append(grad / max(float(grad.max()), 1e-8))
    weights = torch.as_tensor(np.stack(maps)[:, None], device=masks.device, dtype=torch.float32)
    return torch.nn.functional.interpolate(weights, size=size, mode='bilinear', align_corners=False)


def complexity_from_native_labels(he, labels, pixel_size_um=0.5, grid_size=16):
    """Methods 4 score from a registered native-resolution RGB image and cell labels.

    ``labels`` is an existing instance segmentation, with background zero and a
    distinct positive integer for each cell. This function does not create masks.
    """
    he, labels = np.asarray(he), np.asarray(labels)
    if he.ndim != 3 or he.shape[-1] != 3 or labels.shape != he.shape[:2]:
        raise ValueError('Expected registered RGB [H,W,3] and cell labels [H,W]')
    if labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError('Cell labels must be nonnegative integers')
    h, w = labels.shape
    if h != w or h % grid_size or pixel_size_um <= 0:
        raise ValueError('Expected a square field divisible into the cellular grid')
    image = he.astype(np.float32)
    if image.max() > 1:
        image /= 255.0
    hematoxylin = rgb2hed(np.clip(image, 0, 1))[..., 0]
    low, high = float(hematoxylin.min()), float(hematoxylin.max()) + 1e-8
    descriptors, locations = [], []
    patch_pixels = h // grid_size
    for prop in regionprops(labels):
        region = prop.image
        stain = hematoxylin[prop.slice][region]
        hist, _ = np.histogram(stain, bins=32, range=(low, high))
        prob = hist[hist > 0] / max(hist.sum(), 1)
        descriptors.append((prop.eccentricity, prop.solidity,
            min(4 * math.pi * prop.area / max(prop.perimeter ** 2, 1), 1),
            math.log(max(prop.area * pixel_size_um ** 2, 1e-8)),
            prop.equivalent_diameter_area * pixel_size_um,
            -(prob * np.log2(prob)).sum(), stain.var()))
        locations.append((min(int(prop.centroid[0]) // patch_pixels, grid_size - 1),
                          min(int(prop.centroid[1]) // patch_pixels, grid_size - 1)))
    score = np.zeros((grid_size, grid_size), dtype=np.float32)
    if not descriptors:
        return score
    desc = np.asarray(descriptors, dtype=np.float64)
    std = desc.std(0)
    z = np.abs((desc - desc.mean(0)) / np.where(std > 1e-8, std, 1.0))
    locations = np.asarray(locations)
    nmax = (patch_pixels * pixel_size_um) ** 2 / (math.pi * 3 ** 2)
    for r in range(grid_size):
        for c in range(grid_size):
            idx = (locations[:, 0] == r) & (locations[:, 1] == c)
            if not idx.any():
                continue
            delta = np.array((z[idx, :3].mean(), z[idx, 3:5].mean(), z[idx, 5:].mean()))
            heterogeneity = (delta / (1 + delta)).mean()
            score[r, c] = 0.5 * min(idx.sum() / nmax, 1) + 0.5 * heterogeneity
    return score
