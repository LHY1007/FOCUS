"""Build the shared scRNA-seq reference that the RCTD runs use.

Two mouse bone-marrow atlases are reduced to the metadata RCTD needs and mapped
onto the 13-class coarse scheme used by the manuscript figures:

    xiao/young_scrna_anno.h5ad        -> final_celltypes
    xiao/YBM_anno_add_adipoMSC.h5ad   -> celltypes_refined

Two filters are applied. Enucleated cells (RBC, platelets) are dropped because
the spatial data is nuclear segmentation and cannot resolve them. Fine types
with 25 or fewer cells are dropped to satisfy the ``cell_min`` requirement of
spacexr / rctd-py, which is why classes such as ``Lepr_CAR`` and
``Chondrocytes`` are absent from the reference and can never be called.

Writes ``Reference_MetaData_cleaned.csv.gz`` and ``bone_labels.csv``.
"""

from pathlib import Path

import anndata as ad
import pandas as pd

BASE = Path("/data_temp/mnt/raid5/qnf/bone_project/project_bone")
OUTPUT_DIR = BASE / "downstream/deconvolution/output"

# 参考来源
# 想加 T2DM 参考就再加一行（标签列 refined_celltypes）；NC 是正常样本，默认不用
SOURCES = [
    dict(name="young", h5ad=BASE / "xiao/young_scrna_anno.h5ad",
         label_col="final_celltypes"),
    dict(name="YBM", h5ad=BASE / "xiao/YBM_anno_add_adipoMSC.h5ad",
         label_col="celltypes_refined"),
]

# 标签映射
# fine -> coarse，13 类骨头粗粒度体系（与项目已有图件保持一致）
FINE_TO_COARSE = {
    # 骨
    "Osteoblast": "Osteoblast",
    "Osteoclast": "Osteoclast",
    "Lepr_MSC": "LepR+MSC",
    "Lepr_CAR": "LepR+MSC",
    "Fibro-MSC": "Fibro-MSC",
    "MSC": "Fibro-MSC",
    "Tenocyte": "Fibro-MSC",       # 纤维性基质细胞，粗粒度并入 Fibro-MSC
    "Adipo_MSC": "Adipo-MSC",
    "Chondrocytes": "Chondrocytes",
    # 血管
    "AECs": "Endothelial",
    "SECs": "Endothelial",
    "Endo_Other": "Endothelial",
    "Endo_Sinusoidal": "Endothelial",
    "Endo_Arterial": "Endothelial",
    "Endo_Lymph": "Endothelial",
    "Endo_HSC_like": "Endothelial",
    "SMCs": "Vascular_SMC",
    "VSMC": "Vascular_SMC",
    # 淋巴系
    "B_cells": "Lymphoid", "B_pre": "Lymphoid", "B_pro": "Lymphoid",
    "B_mature": "Lymphoid", "B_trans": "Lymphoid", "B_memory": "Lymphoid",
    "B_activated": "Lymphoid", "B_pro_prolif": "Lymphoid",
    "Plasma_cells": "Lymphoid", "Plasma": "Lymphoid",
    "T_CD4": "Lymphoid", "T_CD8": "Lymphoid", "T_Naive": "Lymphoid",
    "T_Cd4": "Lymphoid", "T_Cd8": "Lymphoid", "TNK": "Lymphoid",
    "NK": "Lymphoid", "ILC2": "Lymphoid",
    # HSPC
    "HSPC": "HSPC", "HSC": "HSPC", "STHSC": "HSPC",
    "MPP2": "HSPC", "MPP3": "HSPC", "MPP4": "HSPC",
    "FcG_neg_MPP3": "HSPC", "FcG_pos_MPP3": "HSPC",
    "GMP": "HSPC", "CLP": "HSPC", "MkP": "HSPC",
    "EryP": "HSPC", "Megakaryocyte": "HSPC",
    # 髓系
    "Neu": "Myeloid", "Neu_pre": "Myeloid", "Mono/Macro": "Myeloid",
    "Macro": "Myeloid", "cDC": "Myeloid", "pDC": "Myeloid",
    "Basophil": "Myeloid", "Eosinophil": "Myeloid", "Mast": "Myeloid",
}

# 无核细胞：核分割的空间数据里检不到，参考里剔除（映射表里也就不需要它们）
DROP_FINE = {"RBC", "Platelets"}

MIN_CELLS = 25  # rctd-py / spacexr 的 cell_min

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for src in SOURCES:
        a = ad.read_h5ad(src["h5ad"], backed="r")
        obs = a.obs[[src["label_col"]]].copy()
        obs.columns = ["fine"]
        obs["fine"] = obs["fine"].astype(str)
        obs["source"] = src["name"]
        obs["barcode"] = obs.index.astype(str)
        frames.append(obs.reset_index(drop=True))
        print(f"{src['name']:6s}: {len(obs)} cells, "
              f"{obs['fine'].nunique()} fine types  ({src['h5ad'].name})")

    meta = pd.concat(frames, ignore_index=True)

    # 无核细胞剔除
    dropped_nuc = meta["fine"].isin(DROP_FINE)
    print(f"\n无核细胞剔除: {int(dropped_nuc.sum())} cells "
          f"({meta.loc[dropped_nuc, 'fine'].value_counts().to_dict()})")
    meta = meta[~dropped_nuc].copy()

    # fine -> coarse（未覆盖的标签直接报错，不静默吞掉）
    unknown = sorted(set(meta["fine"]) - set(FINE_TO_COARSE))
    if unknown:
        raise ValueError(f"fine 标签未覆盖，请补 FINE_TO_COARSE: {unknown}")
    meta["coarse"] = meta["fine"].map(FINE_TO_COARSE)

    # <=25 细胞的 fine 类型剔除
    counts = meta["fine"].value_counts()
    dropped = counts[counts <= MIN_CELLS]
    if len(dropped):
        print(f"\nfine 类型 <= {MIN_CELLS} 细胞被剔除: {dropped.to_dict()}")
    meta = meta[meta["fine"].isin(counts[counts > MIN_CELLS].index)].copy()

    # 每来源内 barcode 必须唯一（02 会按 source+barcode 取数）
    for name, g in meta.groupby("source"):
        assert g["barcode"].is_unique, f"{name} 内 barcode 重复"

    print(f"\n最终参考: {len(meta)} cells, {meta['fine'].nunique()} fine types, "
          f"{meta['coarse'].nunique()} coarse types")
    print("\ncoarse x source (cells):")
    print(pd.crosstab(meta["coarse"], meta["source"]).to_string())
    print("\nfine 类型清单:")
    print(meta["fine"].value_counts().to_string())

    # 输出
    out_meta = meta[["barcode", "source", "fine", "coarse"]].rename(
        columns={"fine": "CellType_fine", "coarse": "CellType_coarse"})
    out_meta.to_csv(OUTPUT_DIR / "Reference_MetaData_cleaned.csv.gz",
                    index=False, compression="gzip")
    print(f"\nSaved: {OUTPUT_DIR / 'Reference_MetaData_cleaned.csv.gz'}")

    labels = (meta.groupby("fine")
              .agg(coarse=("coarse", "first"),
                   n=("barcode", "size"),
                   n_young=("source", lambda s: (s == "young").sum()),
                   n_YBM=("source", lambda s: (s == "YBM").sum()))
              .rename_axis("fine").sort_values(["coarse", "fine"]))
    labels.to_csv(OUTPUT_DIR / "bone_labels.csv")
    print(f"Saved: {OUTPUT_DIR / 'bone_labels.csv'}")

if __name__ == "__main__":
    main()
