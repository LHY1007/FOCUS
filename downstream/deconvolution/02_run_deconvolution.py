"""Deconvolve the spatial sample against the reference built by step 01.

Runs rctd-py in ``doublet`` mode: the scRNA-seq atlases are combined into one
joint reference, which is then fitted to the spatial data. Each spot receives a
class, two label levels, and a proportion vector over both the fine and the
coarse type set.

The spatial input is either a cell-level ``.h5ad`` (the bin2cell output, and the
default) or a Space Ranger ``square_008um`` bin directory; ``SPATIAL_SAMPLES``
accepts both. Space Ranger ``scalefactors_json.json`` is used to convert barcode
positions into the array coordinate frame.

Writes, per sample:

    DeconvolutionResults_<name>.csv.gz   barcode, class, label1, label2
    <name>_deconvolved.h5ad              counts, class/label obs, fine and
                                         coarse proportion matrices, coordinates

A sample whose outputs already exist is skipped, so the script can be rerun.
"""

import json
import os
import time
from pathlib import Path

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TRITON_LIBCUDA_PATH"] = "/usr/lib/x86_64-linux-gnu"

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from rctd import Reference, RCTDConfig, run_rctd
from rctd._types import SPOT_CLASS_NAMES

BASE = Path("/data_temp/mnt/raid5/qnf/bone_project/project_bone")
RUN_DIR = BASE / "downstream/deconvolution"
OUTPUT_DIR = RUN_DIR / "output"
REF_META = OUTPUT_DIR / "Reference_MetaData_cleaned.csv.gz"
LABELS = OUTPUT_DIR / "bone_labels.csv"

# 参考 h5ad（01 用的两份，02 从这里取 counts）
REF_H5ADS = {
    "young": BASE / "xiao/young_scrna_anno.h5ad",
    "YBM": BASE / "xiao/YBM_anno_add_adipoMSC.h5ad",
}

# 空间输入：细胞级 h5ad（默认）；换成 bin 目录也可以，两者都支持
SPATIAL_SAMPLES = {
    "16-4-NC": BASE / "xiao/16-4-NC_SR.h5ad",
    # "16-4-NC": BASE / "ST/AXB-1-2-A1/16-4-NC_SR/binned_outputs/square_008um",
}

# coarse 输出列序（参考里没有的类型不会出现，例如 Osteocytes / Adipocytes）
MAJOR_ORDER = ["Osteoblast", "Osteoclast", "Osteocytes", "LepR+MSC", "Fibro-MSC",
               "Adipo-MSC", "Adipocytes", "Chondrocytes", "Endothelial",
               "Vascular_SMC", "Lymphoid", "HSPC", "Myeloid"]

def ensure_raw_counts(a: ad.AnnData, name: str) -> ad.AnnData:
    """RCTD 要求原始 counts；非整数（log-normalized）时 expm1 还原（有损，打印告警）。"""
    X = a.X
    d = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if np.allclose(d, np.round(d)):
        print(f"  {name}: X 已是原始 counts (max={float(d.max()):.0f})")
    else:
        print(f"  !! {name}: X 非整数（log-normalized, max={float(d.max()):.2f}）——"
              f"expm1 还原，有损；原始 counts 不可得时只能这样")
        if sp.issparse(X):
            X = X.copy()
            X.data = np.round(np.expm1(X.data))
        else:
            X = np.round(np.expm1(X))
    a.X = csr_matrix(X, dtype=np.float32) if not sp.issparse(X) else X.astype(np.float32).tocsr()
    return a

def build_reference() -> ad.AnnData:
    """读 01 的元数据，按 source 从各自 h5ad 取 counts，拼成统一参考。"""
    meta = pd.read_csv(REF_META)
    print(f"参考元数据: {len(meta)} cells "
          f"({meta['source'].value_counts().to_dict()}), "
          f"{meta['CellType_fine'].nunique()} fine types")

    parts = []
    for src, path in REF_H5ADS.items():
        m = meta[meta["source"] == src]
        if m.empty:
            continue
        a = ad.read_h5ad(path)
        a = a[m["barcode"].values].copy()          # h5ad 索引即 barcode
        a = ensure_raw_counts(a, src)
        a.obs["cell_type"] = m.set_index("barcode").loc[a.obs_names, "CellType_fine"].values
        parts.append(a)
        print(f"  {src}: {a.n_obs} cells x {a.n_vars} genes")

    ref = ad.concat(parts, join="inner", label=None) if len(parts) > 1 else parts[0]
    print(f"  合并后（共同基因）: {ref.n_obs} cells x {ref.n_vars} genes, "
          f"{ref.obs['cell_type'].nunique()} fine types")
    return ref

def load_spatial(path: Path) -> ad.AnnData:
    """支持两种输入：细胞级 h5ad（默认）或 Space Ranger binned_outputs 目录。"""
    if path.is_dir():                              # Space Ranger bin 目录
        import pyarrow.parquet as pq
        h5 = path / "filtered_feature_bc_matrix.h5"
        parquet = path / "spatial" / "tissue_positions.parquet"
        adata = sc.read_10x_h5(str(h5))
        adata.var_names_make_unique()
        coords = pq.read_table(str(parquet)).to_pandas().set_index("barcode")
        coords = coords.loc[adata.obs_names]
        adata.obsm["spatial"] = coords[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
        adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        sf = path / "spatial" / "scalefactors_json.json"
        if sf.exists():
            adata.uns["scalefactors"] = json.loads(sf.read_text())
        src = f"bins {path.name}"
    else:                                          # 细胞级 h5ad
        adata = sc.read_h5ad(path)
        if "spatial" not in adata.obsm:
            raise ValueError(f"{path} 没有 obsm['spatial']，无法定位")
        if "total_counts" not in adata.obs:
            adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        src = "cell-level h5ad"

    print(f"  {src}: {adata.n_obs} x {adata.n_vars}, "
          f"中位 counts {int(np.median(adata.obs['total_counts']))}, "
          f"0-count {int((adata.obs['total_counts'] == 0).sum())}")
    return adata

def export(adata: ad.AnnData, result, name: str, fine2major: dict):
    """写 CSV（4 列）+ h5ad（label、proportion(fine+coarse)、坐标）。

    rctd-py 语义（与 adhesion 版一致，已在 rctd 源码 _doublet.py/_rctd.py 核对）：
    weights / spot_class / first_type / second_type 都是"被处理 bin"一行（长度 n_kept），
    pixel_mask 是 (N_total,) 把处理过的行映射回原位；未处理的 bin 为 "NA"、proportion 全 0。
    """
    ct = np.array(result.cell_type_names)
    N = adata.n_obs
    kept = (np.where(result.pixel_mask)[0] if result.pixel_mask is not None
            else np.arange(N))
    n_kept = len(kept)
    assert result.pixel_mask is None or result.pixel_mask.shape[0] == N
    assert result.weights.shape == (n_kept, len(ct)), (result.weights.shape, n_kept, len(ct))
    assert len(result.spot_class) == n_kept

    cls = np.full(N, "NA", dtype=object)
    lab1 = np.full(N, "NA", dtype=object)
    lab2 = np.full(N, "NA", dtype=object)
    processed = np.zeros(N, dtype=bool)
    processed[kept] = True
    cls[kept] = np.asarray(SPOT_CLASS_NAMES, dtype=object)[np.asarray(result.spot_class)]
    lab1[kept] = ct[np.asarray(result.first_type)]
    lab2[kept] = ct[np.asarray(result.second_type)]

    # 4 列 CSV
    df = pd.DataFrame({"barcode": adata.obs_names,
                       "DeconvolutionClass": cls,
                       "DeconvolutionLabel1": lab1,
                       "DeconvolutionLabel2": lab2})
    df.to_csv(OUTPUT_DIR / f"DeconvolutionResults_{name}.csv.gz",
              index=False, compression="gzip")
    print(f"  Saved: DeconvolutionResults_{name}.csv.gz  "
          f"{df['DeconvolutionClass'].value_counts().to_dict()}")

    # proportion（fine）：全长度，未处理的位置为 0
    W_arr = np.zeros((N, len(ct)), dtype=np.float32)
    W_arr[kept] = np.asarray(result.weights, dtype=np.float32)
    W = pd.DataFrame(W_arr, index=adata.obs_names, columns=ct)

    # proportion（coarse）= 同一 coarse 的 fine 列求和
    groups = pd.Series([fine2major[t] for t in ct], index=ct)
    present_major = [m for m in MAJOR_ORDER if m in set(groups.values)]
    W_major = (W.T.groupby(groups).sum().T).reindex(columns=present_major, fill_value=0.0)

    # obs
    s1, s2 = pd.Series(lab1), pd.Series(lab2)
    adata.obs["DeconvolutionClass"] = pd.Categorical(cls)
    adata.obs["Label1_fine"] = pd.Categorical(s1)
    adata.obs["Label2_fine"] = pd.Categorical(s2)
    adata.obs["Label1_major"] = pd.Categorical(s1.map(fine2major).fillna("NA"))
    adata.obs["Label2_major"] = pd.Categorical(s2.map(fine2major).fillna("NA"))
    adata.obs["rctd_processed"] = processed
    adata.obsm["proportions"] = W
    adata.obsm["proportions_major"] = W_major

    h5ad = OUTPUT_DIR / f"{name}_deconvolved.h5ad"
    adata.write_h5ad(h5ad, compression="gzip")
    print(f"  Saved: {h5ad.name}")

    comp = W_major.loc[processed].mean(axis=0).sort_values(ascending=False)
    print("  mean coarse proportions (processed cells): "
          + ", ".join(f"{k} {v:.3f}" for k, v in comp.head(8).items()))

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)   # 日志按行刷新
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fine2major = pd.read_csv(LABELS).set_index("fine")["coarse"].to_dict()
    missing = set(pd.read_csv(REF_META)["CellType_fine"].unique()) - set(fine2major)
    if missing:
        raise ValueError(f"bone_labels.csv 缺这些 fine 类型: {missing}")

    ref_adata = build_reference()
    print("\n构建 rctd-py Reference...")
    reference = Reference(ref_adata, cell_type_col="cell_type", cell_min=25)
    print(f"  {reference.n_genes} genes, {reference.n_types} types")
    del ref_adata

    for name, path in SPATIAL_SAMPLES.items():
        csv = OUTPUT_DIR / f"DeconvolutionResults_{name}.csv.gz"
        h5ad = OUTPUT_DIR / f"{name}_deconvolved.h5ad"
        if csv.exists() and h5ad.exists():
            print(f"\n=== {name}: SKIP（输出已存在）===")
            continue

        print(f"\n=== {name} ===")
        t0 = time.time()
        spatial = load_spatial(path)
        config = RCTDConfig(device="auto", dtype="float32", compile=False)
        result = run_rctd(spatial=spatial, reference=reference, mode="doublet",
                          config=config, batch_size=5000)
        print(f"  RCTD done in {time.time() - t0:.0f}s")
        export(spatial, result, name, fine2major)
        del spatial, result

    print("\nALL SAMPLES DONE")

if __name__ == "__main__":
    main()
