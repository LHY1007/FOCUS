"""Extract one registered Xenium H&E/ST tile from the original 10x files.

Local files and HTTP range-readable files are accepted. Only Parquet row groups
whose coordinate statistics intersect the requested field and OME-TIFF tiles
needed by the registered image crop are decoded.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import fsspec
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
import tifffile


FIELD_UM = 2560
TILE_PX = 5120
TARGET_MPP = 0.5


def _open(path):
    return fsspec.open(str(path), "rb", block_size=1 << 20, cache_type="bytes")


def _panel(path):
    targets = json.loads(Path(path).read_text())["payload"]["targets"]
    names = [target["type"]["data"]["name"] for target in targets]
    if len(names) != len(set(names)):
        raise ValueError("The Xenium panel contains duplicate feature names")
    return names


def _counts(parquet_path, names, x0, y0):
    x1, y1 = x0 + FIELD_UM, y0 + FIELD_UM
    columns = ["feature_name", "x_location", "y_location"]
    gene_map = {name: i for i, name in enumerate(names)}
    hr = sparse.csr_matrix((256 * 256, len(names)), dtype=np.int32)
    inspected = selected = matched = 0
    with _open(parquet_path) as source:
        parquet = pq.ParquetFile(source)
        if not set(columns).issubset(parquet.schema.names):
            raise ValueError("Transcript file is missing required Xenium columns")
        for index in range(parquet.num_row_groups):
            group = parquet.metadata.row_group(index)
            stats = {group.column(j).path_in_schema: group.column(j).statistics
                     for j in range(group.num_columns)}
            if any(stats.get(axis) is None or not stats[axis].has_min_max
                   for axis in ("x_location", "y_location")):
                raise ValueError("Parquet row group lacks coordinate statistics")
            xs, ys = stats["x_location"], stats["y_location"]
            if xs.max < x0 or xs.min >= x1 or ys.max < y0 or ys.min >= y1:
                continue
            inspected += 1
            table = parquet.read_row_group(index, columns=columns)
            x = table["x_location"].to_numpy(zero_copy_only=False)
            y = table["y_location"].to_numpy(zero_copy_only=False)
            inside = np.isfinite(x) & np.isfinite(y) & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            if not inside.any():
                continue
            x, y = x[inside], y[inside]
            selected += len(x)
            symbol = np.asarray(table["feature_name"].to_pylist(), dtype=object)[inside]
            gene = np.fromiter((gene_map.get(value, -1) for value in symbol), dtype=np.int32)
            keep = gene >= 0
            matched += int(keep.sum())
            row = (np.floor((y[keep] - y0) / 10).astype(np.int32) * 256
                   + np.floor((x[keep] - x0) / 10).astype(np.int32))
            chunk = sparse.coo_matrix((np.ones(int(keep.sum()), dtype=np.int32),
                                       (row, gene[keep])), shape=hr.shape).tocsr()
            hr += chunk
            print(f"Parquet row group {index}: {int(keep.sum())} panel transcripts", flush=True)
    # Row mappings retain the same 2,560 um field at 20 and 100 um pitches.
    yy, xx = np.divmod(np.arange(256 * 256, dtype=np.int32), 256)
    def rebin(side, factor):
        target = (yy // factor) * side + xx // factor
        mapping = sparse.coo_matrix((np.ones(len(target), dtype=np.int32),
                                     (target, np.arange(len(target)))),
                                    shape=(side * side, len(target))).tocsr()
        return (mapping @ hr).tocsr()
    hr128 = rebin(128, 2)
    spot = rebin(26, 10)
    if any(int(matrix.sum(dtype=np.int64)) != matched for matrix in (hr, hr128, spot)):
        raise ValueError("Transcript counts differ between resolutions")
    return hr, hr128, spot, {"row_groups_read": inspected,
                             "transcripts_in_field": selected,
                             "panel_transcripts_in_field": matched}


def _he_crop(tiff_path, affine, target_x, target_y):
    inverse = np.linalg.inv(affine)
    corners = np.asarray([[target_x, target_y, 1],
                          [target_x + TILE_PX, target_y, 1],
                          [target_x, target_y + TILE_PX, 1],
                          [target_x + TILE_PX, target_y + TILE_PX, 1]], dtype=np.float64).T
    source_xy = (inverse @ corners)[:2]
    sx0, sy0 = map(int, np.floor(source_xy.min(axis=1) - 2))
    sx1, sy1 = map(int, np.ceil(source_xy.max(axis=1) + 2))
    if sx0 < 0 or sy0 < 0:
        raise ValueError("Requested tile extends beyond the raw H&E image")
    with _open(tiff_path) as source, tifffile.TiffFile(source) as tif:
        page = tif.pages[0]
        if not page.is_tiled or page.dtype != np.uint8 or page.samplesperpixel != 3:
            raise ValueError("Expected tiled RGB uint8 Xenium H&E OME-TIFF")
        if sx1 > page.imagewidth or sy1 > page.imagelength:
            raise ValueError("Requested tile extends beyond the raw H&E image")
        tw, th = page.tilewidth, page.tilelength
        ncols = math.ceil(page.imagewidth / tw)
        roi = np.empty((sy1 - sy0, sx1 - sx0, 3), dtype=np.uint8)
        tile_count = 0
        for ty in range(sy0 // th, math.ceil(sy1 / th)):
            for tx in range(sx0 // tw, math.ceil(sx1 / tw)):
                index = ty * ncols + tx
                source.seek(page.dataoffsets[index])
                payload = source.read(page.databytecounts[index])
                decoded, _, _ = page.decode(payload, index, jpegtables=page.jpegtables)
                tile = decoded[0]
                ax0, ay0 = max(tx * tw, sx0), max(ty * th, sy0)
                ax1, ay1 = min((tx + 1) * tw, sx1), min((ty + 1) * th, sy1)
                roi[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0] = tile[
                    ay0 - ty * th:ay1 - ty * th, ax0 - tx * tw:ax1 - tx * tw]
                tile_count += 1
                if tile_count % 25 == 0:
                    print(f"Decoded {tile_count} H&E TIFF tiles", flush=True)
    transform = affine[:2, :2].copy()
    offset = affine[:2, :2] @ np.asarray([sx0, sy0]) + affine[:2, 2] - [target_x, target_y]
    local = np.column_stack((transform, offset))
    cv2.setNumThreads(1)
    image = cv2.warpAffine(roi, local, (TILE_PX, TILE_PX), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return image, {"source_bbox_px": [sx0, sy0, sx1, sy1], "source_tiff_tiles": tile_count}


def replay(raw_dir, he_path, transcript_path, tile_id, output):
    raw_dir, output = Path(raw_dir), Path(output)
    i, j = (int(part) for part in tile_id.split("_"))
    if i < 0 or j < 0:
        raise ValueError("Tile indices must be non-negative")
    experiment = json.loads((raw_dir / "experiment.xenium").read_text())
    names = _panel(raw_dir / "gene_panel.json")
    affine = np.loadtxt(raw_dir / "he_imagealignment.csv", delimiter=",")
    if affine.shape != (3, 3) or not np.allclose(affine[2], [0, 0, 1]):
        raise ValueError("Expected a 3-by-3 H&E to Xenium morphology transform")
    pixel_size = float(experiment["pixel_size"])
    affine[:2] *= pixel_size / TARGET_MPP
    x0, y0 = i * FIELD_UM // 2, j * FIELD_UM // 2
    hr, hr128, spot, counts_info = _counts(transcript_path, names, x0, y0)
    image, image_info = _he_crop(he_path, affine, i * TILE_PX // 2, j * TILE_PX // 2)
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "HR_ST_256.npz", hr)
    sparse.save_npz(output / "HR_ST_128.npz", hr128)
    sparse.save_npz(output / "spot_ST.npz", spot)
    np.save(output / "5120.npy", image)
    np.save(output / "5120_to256.npy", cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR))
    with (output / "gene_names.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Gene_Name"])
        writer.writerows((name,) for name in names)
    record = {"tile_id": tile_id, "x_um": x0, "y_um": y0, "field_um": FIELD_UM,
              "original_st_bin_um": 10.0, "source_pixel_um": pixel_size,
              "he_source": str(he_path), "transcript_source": str(transcript_path),
              "panel_sha256": hashlib.sha256((raw_dir / "gene_panel.json").read_bytes()).hexdigest(),
              "alignment_sha256": hashlib.sha256((raw_dir / "he_imagealignment.csv").read_bytes()).hexdigest(),
              "gene_count": len(names), "counts": counts_info, "image": image_info,
              "count_dtype": "int32", "transcript_filter": "panel names and field coordinates; no QV filter"}
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--he", required=True, help="Local OME-TIFF or HTTP range URL")
    parser.add_argument("--transcripts", required=True, help="Local Parquet or HTTP range URL")
    parser.add_argument("--tile", required=True, help="Overlapping tile index, for example 3_10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.raw_dir, args.he, args.transcripts, args.tile, args.output)


if __name__ == "__main__":
    main()
