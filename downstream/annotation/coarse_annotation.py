"""Assign coarse cell-type labels to the NC sample (16-4-NC).

Two label sources are combined. The YBM-reference RCTD result is the primary
source, being a molecular call; where it has no label the geo-labelled
annotation fills the gap. Cells that neither source covers stay ``Unassigned``
rather than being forced into a class.

Writes the annotated AnnData object, a label table, and the spatial cell map.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')

BASE = "/data_temp/mnt/raid5/qnf/bone_project/project_bone"
OUT_DIR = f"{BASE}/downstream/annotation/output"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# Step 1: Load data
print("Loading data...")

ad_sp = sc.read_h5ad(f"{BASE}/xiao/16-4-NC_SR.h5ad")
print(f"  Spatial: {ad_sp.n_obs} cells, {ad_sp.n_vars} genes")

# Primary: xzq YBM-ref RCTD
ybm_csv = pd.read_csv(
    '/data_temp/mnt/raid5/xzq/spatial_bone/project_bone/RCTD/1-2/NC/soj_rctd_first_type_nc_min20.csv',
    index_col=0
)
ybm_csv.index = ybm_csv.index.astype(str)

# Supplementary: xzq geo-labeled
ad_geo = sc.read_h5ad(
    '/data_temp/mnt/raid5/xzq/spatial_bone/project_bone/RCTD/label_spot_by_geo/1-2_NC_ADD_CELLTYPE/4-NC_celltype.h5ad'
)

# Step 2: Fine → coarse mapping (YBM-ref: 44 types → 13 coarse)
FINE_TO_COARSE = {
    # Bone
    'Osteoblast': 'Osteoblast',
    'Osteoclast': 'Osteoclast',
    'Lepr_CAR': 'LepR+MSC',
    'MSC': 'Fibro-MSC',
    'Chondrocytes': 'Chondrocytes',
    # Vascular
    'Endo_Sinusoidal': 'Endothelial',
    'Endo_Lymph': 'Endothelial',
    'Endo_Arterial': 'Endothelial',
    'Endo_Other': 'Endothelial',
    'Endo_HSC_like': 'Endothelial',
    'VSMC': 'Vascular_SMC',
    # Lymphoid
    'B_pro_prolif': 'Lymphoid', 'B_pro': 'Lymphoid', 'B_pre': 'Lymphoid',
    'B_trans': 'Lymphoid', 'B_memory': 'Lymphoid', 'B_mature': 'Lymphoid',
    'B_activated': 'Lymphoid', 'Plasma': 'Lymphoid',
    'T_Cd4': 'Lymphoid', 'T_Cd8': 'Lymphoid', 'TNK': 'Lymphoid',
    'NK': 'Lymphoid', 'ILC2': 'Lymphoid',
    # HSPC
    'HSC': 'HSPC', 'STHSC': 'HSPC',
    'MPP2': 'HSPC', 'MPP3': 'HSPC', 'MPP4': 'HSPC',
    'GMP': 'HSPC', 'CLP': 'HSPC', 'MkP': 'HSPC', 'EryP': 'HSPC',
    # Myeloid
    'Neu_pre': 'Myeloid', 'Neu': 'Myeloid', 'Macro': 'Myeloid',
    'cDC': 'Myeloid', 'pDC': 'Myeloid',
    'Eosinophil': 'Myeloid', 'Basophil': 'Myeloid', 'Mast': 'Myeloid',
    # → Unassigned
    'Megakaryocyte': 'Unassigned',
    'Neuro_Stromal': 'Unassigned',
    'Tenocyte': 'Unassigned',
}

# Also map geo h5ad types → coarse (for cells not covered by YBM-ref)
GEO_TO_COARSE = {
    'Neutrophils': 'Myeloid',
    'B_lineage': 'Lymphoid',
    'Osteoblast': 'Osteoblast',
    'Chondrocytes': 'Chondrocytes',
    'GMP': 'HSPC',
    'Bone_tissue': 'Osteocytes',     # Bone cortex → Osteocytes
    'Endothelial': 'Endothelial',
    'EryP': 'HSPC',
    'Macrophages': 'Myeloid',
    'Megakaryocyte': 'Unassigned',
    'DC': 'Myeloid',
    'MPP': 'HSPC',
    'Lepr_CAR': 'LepR+MSC',
    'Granulocytes': 'Myeloid',
    'MkP': 'HSPC',
    'HSC': 'HSPC',
    'MSC': 'Fibro-MSC',
    'T_NK_ILC': 'Lymphoid',
    'VSMC': 'Vascular_SMC',
    'Mast_cell': 'Myeloid',
    'Neuro_Stromal': 'Unassigned',
    'Tenocyte': 'Unassigned',
    'Osteoclast': 'Osteoclast',
    'Adipocytes': 'Adipocytes',       # Only 31, but needed for coverage
}

# Step 3: Assign coarse types
# Initialize all cells as Unassigned
coarse = pd.Series('Unassigned', index=ad_sp.obs_names)

# 3a. Apply YBM-ref mapping (molecular RCTD, highest confidence)
ybm_types = ybm_csv['x']
ybm_coarse = ybm_types.map(FINE_TO_COARSE)
ybm_coarse = ybm_coarse.dropna()
common_idx = ybm_coarse.index.intersection(ad_sp.obs_names)
coarse.loc[common_idx] = ybm_coarse.loc[common_idx]
print(f"  YBM-ref → {len(common_idx)} cells annotated")

# 3b. Fill remaining with geo-label
geo_types = ad_geo.obs['cell_type']
geo_coarse = geo_types.map(GEO_TO_COARSE)
geo_coarse = geo_coarse.dropna()
still_unassigned = coarse == 'Unassigned'
geo_fill_idx = geo_coarse.index.intersection(ad_sp.obs_names[still_unassigned])
coarse.loc[geo_fill_idx] = geo_coarse.loc[geo_fill_idx]
print(f"  Geo-label → {len(geo_fill_idx)} additional cells")

# Step 4: Save and report
ad_sp.obs['coarse_celltype'] = coarse.values.astype(str)

print(f"\n=== Final coarse cell type distribution ===")
dist = ad_sp.obs['coarse_celltype'].value_counts()
for ct, cnt in dist.items():
    pct = cnt / ad_sp.n_obs * 100
    print(f"  {ct:20s} {cnt:7d}  ({pct:5.1f}%)")

# Check coverage
annotated = (coarse != 'Unassigned').sum()
print(f"\n  Annotated: {annotated}/{ad_sp.n_obs} ({annotated/ad_sp.n_obs*100:.1f}%)")
print(f"  Cell types present: {dist.index.tolist()}")

# Step 5: Visualization
print("\nGenerating visualization...")

COLORS = {
    'Osteoblast':   '#FF0000',
    'Osteoclast':   '#FF8C00',
    'Osteocytes':   '#8B4513',
    'LepR+MSC':     '#FF00FF',
    'Fibro-MSC':    '#FF69B4',
    'Adipocytes':   '#FFD700',
    'Chondrocytes': '#006400',
    'Endothelial':  '#00BFFF',
    'Vascular_SMC': '#008080',
    'Lymphoid':     '#ADFF2F',
    'HSPC':         '#DC143C',
    'Myeloid':      '#191970',
    'Unassigned':   '#E8E8E8',
}

coords = ad_sp.obsm['spatial']
boundaries = ad_sp.obs['boundary'].values
celltypes = ad_sp.obs['coarse_celltype'].values

fig, ax = plt.subplots(1, 1, figsize=(30, 12))
ax.set_facecolor('white')

for batch_start in range(0, len(celltypes), 10000):
    batch_end = min(batch_start + 10000, len(celltypes))
    patches = []
    facecolors = []

    for i in range(batch_start, batch_end):
        boundary = boundaries[i]
        ct = celltypes[i]
        color = COLORS.get(ct, '#E8E8E8')

        if isinstance(boundary, str):
            try:
                boundary = literal_eval(boundary)
            except:
                continue
        if boundary is None or len(boundary) < 3:
            continue

        poly_coords = np.array(boundary)
        polygon = Polygon(poly_coords, closed=True)
        patches.append(polygon)
        facecolors.append(color)

    if patches:
        pc = PatchCollection(patches, facecolors=facecolors, edgecolors='black',
                             alpha=0.8, linewidths=0.15)
        ax.add_collection(pc)

    if (batch_start // 10000) % 5 == 0:
        print(f'  Processed {batch_end}/{len(celltypes)} cells...')

x_min, x_max = coords[:, 0].min() - 50, coords[:, 0].max() + 50
y_min, y_max = coords[:, 1].min() - 50, coords[:, 1].max() + 50
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_max, y_min)
ax.set_aspect('equal')
ax.axis('off')

present_types = set(celltypes)
legend_elements = [Patch(facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8, label=ct)
                   for ct, color in COLORS.items()
                   if ct in present_types and ct != 'Unassigned']
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5),
          fontsize=12, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/coarse_celltype_nc.pdf', bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT_DIR}/coarse_celltype_nc.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved: {OUT_DIR}/coarse_celltype_nc.png')

# Save
ad_sp.write_h5ad(f"{OUT_DIR}/16-4-NC_SR_coarse_annotated.h5ad")
ad_sp.obs[['coarse_celltype']].to_csv(f"{OUT_DIR}/coarse_celltype_nc.csv")
print(f'Saved: {OUT_DIR}/16-4-NC_SR_coarse_annotated.h5ad')
print('Done!')
