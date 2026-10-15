
"""Quality control for the deconvolution output.

Three checks are run against ``<name>_deconvolved.h5ad``:

1. composition -- the class distribution, the share of spots RCTD could assign,
   and the median counts per class;
2. markers -- the detection rate of canonical markers for each coarse class
   (Lepr/Cxcl12 for MSC, Alpl/Sp7 for osteoblasts, Ctsk/Acp5 for osteoclasts,
   and so on). A class whose markers are not enriched over the rest of the
   section is not supported by the data;
3. agreement -- the coarse-label agreement with an independent YBM-reference
   RCTD run. This check is skipped when that file is not present.

The report is printed and written next to the outputs.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE = Path("/data_temp/mnt/raid5/qnf/bone_project/project_bone")
OUT = BASE / "downstream/deconvolution/output"

SAMPLES = ["16-4-NC"]

# xzq 的 YBM-ref RCTD（外部文件；不在就跳过 Gate 3）
XZQ_CSV = Path("/data_temp/mnt/raid5/xzq/spatial_bone/project_bone/RCTD/1-2/NC/"
               "soj_rctd_first_type_nc_min20.csv")

# marker 面板：coarse 标签 -> 期望高表达的基因
MARKERS = {
    "LepR+MSC": ["Lepr", "Cxcl12", "Kitl"],
    "Fibro-MSC": ["Pdgfra", "Col1a1"],
    "Osteoblast": ["Alpl", "Sp7", "Bglap"],
    "Osteoclast": ["Ctsk", "Acp5"],
    "Chondrocytes": ["Sox9", "Col2a1"],
    "Endothelial": ["Pecam1", "Cdh5"],
    "HSPC": ["Kit", "Cd34"],
    "Myeloid": ["Lyz2", "Csf1r"],
    "Lymphoid": ["Cd3e", "Cd79a"],
    "Vascular_SMC": ["Acta2", "Myh11"],
}

# xzq 的 fine 标签 -> 我们的 coarse（与其 nc_coarse_annotation.py 中的表一致）
XZQ_FINE_TO_COARSE = {
    "Osteoblast": "Osteoblast", "Osteoclast": "Osteoclast",
    "Lepr_CAR": "LepR+MSC", "MSC": "Fibro-MSC", "Chondrocytes": "Chondrocytes",
    "Endo_Sinusoidal": "Endothelial", "Endo_Lymph": "Endothelial",
    "Endo_Arterial": "Endothelial", "Endo_Other": "Endothelial",
    "Endo_HSC_like": "Endothelial", "VSMC": "Vascular_SMC",
    "B_pro_prolif": "Lymphoid", "B_pro": "Lymphoid", "B_pre": "Lymphoid",
    "B_trans": "Lymphoid", "B_memory": "Lymphoid", "B_mature": "Lymphoid",
    "B_activated": "Lymphoid", "Plasma": "Lymphoid",
    "T_Cd4": "Lymphoid", "T_Cd8": "Lymphoid", "TNK": "Lymphoid",
    "NK": "Lymphoid", "ILC2": "Lymphoid",
    "HSC": "HSPC", "STHSC": "HSPC", "MPP2": "HSPC", "MPP3": "HSPC",
    "MPP4": "HSPC", "GMP": "HSPC", "CLP": "HSPC", "MkP": "HSPC", "EryP": "HSPC",
    "Neu_pre": "Myeloid", "Neu": "Myeloid", "Macro": "Myeloid",
    "cDC": "Myeloid", "pDC": "Myeloid", "Eosinophil": "Myeloid",
    "Basophil": "Myeloid", "Mast": "Myeloid",
    "Megakaryocyte": "Unassigned", "Neuro_Stromal": "Unassigned",
    "Tenocyte": "Unassigned",
}

LINES = []

def p(s=""):
    print(s)
    LINES.append(s)

# Gate 1
p("=" * 78)
p("Gate 1: 基本分布（来自 deconvolved h5ad）")
p("=" * 78)

rows = []
for name in SAMPLES:
    f = OUT / f"{name}_deconvolved.h5ad"
    if not f.exists():
        p(f"!! {name}: h5ad 不存在，跳过（先跑 02_run_deconvolution.py）")
        continue
    a = ad.read_h5ad(f)
    cls = a.obs["DeconvolutionClass"].astype(str)
    proc = a.obs["rctd_processed"].values.astype(bool)
    tot = a.obs["total_counts"].values
    n_proc = int(proc.sum())
    rows.append(dict(
        sample=name, cells=len(cls), processed=n_proc,
        na_frac=1 - n_proc / len(cls),
        singlet_frac=(cls == "singlet").sum() / n_proc,
        doublet_certain_frac=(cls == "doublet_certain").sum() / n_proc,
        doublet_uncertain_frac=(cls == "doublet_uncertain").sum() / n_proc,
        reject_frac=(cls == "reject").sum() / n_proc,
        median_counts_proc=float(np.median(tot[proc])),
    ))

per_sample = pd.DataFrame(rows)
if len(per_sample):
    per_sample.to_csv(OUT / "qc_per_sample.csv", index=False)
    p(per_sample.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    p()
    p("判读参考：NA 比例高 = 空细胞多（正常，本样本有 2 万空细胞）；"
      "reject 高 = 参考覆盖不足；doublet 高 = 细胞分割重叠多")

# Gate 2
p()
p("=" * 78)
p("Gate 2: canonical marker 检出率（按 coarse 标签）")
p("=" * 78)

for name in SAMPLES:
    f = OUT / f"{name}_deconvolved.h5ad"
    if not f.exists():
        continue
    a = ad.read_h5ad(f)
    X = a.X
    lab = a.obs["Label1_major"].astype(str).values
    hits = []
    for ct, genes in MARKERS.items():
        m = lab == ct
        if m.sum() == 0:
            continue
        for g in genes:
            if g not in a.var_names:
                continue
            gi = a.var_names.get_loc(g)
            col = X[:, gi]
            col = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
            hits.append(dict(label=ct, gene=g, n_cells=int(m.sum()),
                             pct_positive=100 * float((col[m] > 0).mean()),
                             mean_counts=float(col[m].mean())))
    h = pd.DataFrame(hits)
    if h.empty:
        continue
    p(f"\n{name}:")
    p(h.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    h.to_csv(OUT / f"qc_markers_{name}.csv", index=False)
    p("\n判读参考：每个基因应在同名标签里检出率最高；"
      "若 Lepr/Cxcl12 的最高检出不在 LepR+MSC 里，说明该标签不可信")

# Gate 3
p()
p("=" * 78)
p("Gate 3: vs xzq 的 YBM-ref RCTD（不同实现/参考，比 coarse 一致率）")
p("=" * 78)

for name in SAMPLES:
    f = OUT / f"{name}_deconvolved.h5ad"
    if not (f.exists() and XZQ_CSV.exists()):
        p(f"!! {name}: 缺文件（本方 h5ad 或 xzq csv），跳过")
        continue
    a = ad.read_h5ad(f, backed="r")
    ours = pd.DataFrame({
        "cell_id": a.obs_names.astype(str),
        "ours_class": a.obs["DeconvolutionClass"].astype(str).values,
        "ours_major": a.obs["Label1_major"].astype(str).values,
    })
    xzq = pd.read_csv(XZQ_CSV, index_col=0)
    xzq.index = xzq.index.astype(str)
    xzq.columns = ["xzq_fine"]
    xzq["xzq_major"] = xzq["xzq_fine"].map(XZQ_FINE_TO_COARSE)

    m = ours.merge(xzq, left_on="cell_id", right_index=True, how="inner")
    both = (m["ours_class"] != "NA") & m["xzq_major"].notna()
    agree = (m.loc[both, "ours_major"] == m.loc[both, "xzq_major"]).mean()

    out = pd.DataFrame([dict(
        sample=name, merged_cells=len(m), both_called=int(both.sum()),
        coarse_agree=float(agree),
        ours_processed_frac=float((m["ours_class"] != "NA").mean()),
        xzq_called_frac=float(m["xzq_major"].notna().mean()),
    )])
    out.to_csv(OUT / f"qc_vs_xzq_{name}.csv", index=False)
    p(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    conf = (pd.crosstab(m.loc[both, "ours_major"], m.loc[both, "xzq_major"])
            .sort_values(by=list(pd.crosstab(m.loc[both, "ours_major"],
                                             m.loc[both, "xzq_major"]).columns)[0],
                         ascending=False))
    p("\ncoarse 混淆（行=本方，列=xzq）:")
    p(conf.to_string())
    p("\n判读参考：两种实现/参考不一致是正常的；重点看 (1) 骨相关类型"
      "（Osteoblast/LepR+MSC/Fibro-MSC）是否互相落在对方同一类里，"
      "(2) 本方没有的类（Unassigned）是否被系统性错分")

(OUT / "qc_report.txt").write_text("\n".join(LINES) + "\n")
p()
p(f"Saved: {OUT / 'qc_report.txt'}")
