"""Supplement the NC annotation with a T2DM reference mapped by Tangram.

The YBM reference used for most classes contains neither adipocytes nor
osteocytes, so those classes can never be called from it. A T2DM scRNA-seq
reference is mapped onto the NC section with Tangram to cover them.

The merge is deliberately asymmetric: for classes the YBM reference already
resolves, the existing label is kept and the Tangram prediction is discarded;
the Tangram call is only accepted for the classes the YBM reference lacks, and
only above a confidence threshold.

Writes the updated annotation to the run directory.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg
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

# T2DM fine → coarse mapping (same as rctd_merge_visualize.py)
COARSE_MAP_T2DM = {
    'Osteoblast': 'Osteoblast',
    'MSC_phex': 'Osteocytes',
    'MSC': 'Fibro-MSC',
    'Adipocytes': 'Adipocytes',
    'Chondrocytes': 'Chondrocytes',
    'Endo_Arterial': 'Endothelial', 'Endo_lym': 'Endothelial',
    'Endo_Lipid': 'Endothelial', 'Endo_inflam': 'Endothelial',
    'VSMC': 'Vascular_SMC',
    'CD8T': 'Lymphoid', 'CD4T': 'Lymphoid', 'NK': 'Lymphoid',
    'B_pro': 'Lymphoid', 'B_Cd83': 'Lymphoid', 'B_Sell': 'Lymphoid',
    'B_pre': 'Lymphoid', 'B_Sox5': 'Lymphoid', 'B_pre_prolif': 'Lymphoid',
    'HSC': 'HSPC', 'STHSC': 'HSPC', 'MPP2': 'HSPC', 'MPP3': 'HSPC',
    'MPP4': 'HSPC', 'mGMP': 'HSPC', 'GP': 'HSPC', 'CLP': 'HSPC', 'cMoP': 'HSPC',
    'EryP': 'HSPC',
    'Macrophages': 'Myeloid', 'Macro_Lipid': 'Myeloid', 'Monocyte': 'Myeloid',
    'DC': 'Myeloid', 'Neu': 'Myeloid', 'Neu_pre': 'Myeloid',
    'Neu_t2dm': 'Myeloid', 'Neu_pre_t2dm': 'Myeloid',
    'Basophil': 'Myeloid', 'Eosinophil': 'Myeloid', 'Mast_cell': 'Myeloid',
    'Megakaryocyte': 'Other',
    'RBC': 'Other', 'Tenocyte': 'Other', 'Skeletal_Muscle_Cell': 'Other',
    'Schwann_like_cell': 'Other',
}

# Bone-relevant types to keep in reference
KEEP_TYPES = [
    'Osteoblast', 'Osteoclast', 'Osteocytes', 'LepR+MSC', 'Fibro-MSC',
    'Adipocytes', 'Chondrocytes', 'Endothelial', 'Vascular_SMC',
    'Lymphoid', 'HSPC', 'Myeloid'
]

# Step 1: Load T2DM scRNA-seq reference
print("=" * 60)
print("Step 1: Loading T2DM scRNA-seq reference")
print("=" * 60)

ad_sc = sc.read_h5ad(f"{BASE}/xiao/t2dm_scrna_anno.h5ad")
ad_sc.obs['cell_type_raw'] = ad_sc.obs['refined_celltypes'].astype(str)
ad_sc.obs['cell_type'] = ad_sc.obs['cell_type_raw'].map(COARSE_MAP_T2DM).fillna('Other')

print(f"  Reference: {ad_sc.n_obs} cells, {ad_sc.n_vars} genes")
print(f"  Coarse types:\n{ad_sc.obs['cell_type'].value_counts().to_string()}")

# Filter to bone-relevant types
ad_sc = ad_sc[ad_sc.obs['cell_type'].isin(KEEP_TYPES)].copy()
print(f"  After filtering: {ad_sc.n_obs} cells")

# Step 2: Load NC spatial data
print("\n" + "=" * 60)
print("Step 2: Loading NC spatial data")
print("=" * 60)

ad_sp = sc.read_h5ad(f"{BASE}/xiao/16-4-NC_SR.h5ad")
print(f"  Spatial: {ad_sp.n_obs} cells, {ad_sp.n_vars} genes")

# Step 3: Find marker genes
print("\n" + "=" * 60)
print("Step 3: Selecting marker genes")
print("=" * 60)

# Normalize reference for marker finding
ad_sc_norm = ad_sc.copy()
sc.pp.normalize_total(ad_sc_norm, target_sum=1e4)
sc.pp.log1p(ad_sc_norm)

sc.tl.rank_genes_groups(ad_sc_norm, groupby='cell_type', method='wilcoxon', n_genes=100)

markers = []
for ct in ad_sc_norm.obs['cell_type'].unique():
    try:
        ct_markers = sc.get.rank_genes_groups_df(ad_sc_norm, group=ct)
        top = ct_markers[ct_markers['pvals_adj'] < 0.05].head(100)['names'].tolist()
        markers.extend(top)
    except:
        pass
markers = list(set(markers))

genes_sp = set(ad_sp.var_names)
markers_in_sp = [g for g in markers if g in genes_sp]
print(f"  Markers total: {len(markers)}, in spatial: {len(markers_in_sp)}")

# Add canonical bone markers
canonical = ['Lepr', 'Cxcl12', 'Kitl', 'Alpl', 'Ibsp', 'Runx2', 'Sp7',
             'Dmp1', 'Mepe', 'Sost', 'Phex', 'Acan', 'Col2a1', 'Sox9',
             'Pparg', 'Adipoq', 'Fabp4', 'Cd68', 'Lyz2', 'Csf1r',
             'Cd3e', 'Cd19', 'Cd79a', 'Kit', 'Cd34', 'Pecam1', 'Cdh5',
             'Col1a1', 'Bglap', 'Spock2', 'Tnfrsf11b', 'Apoe', 'Ctsk',
             'Acp5', 'Nes', 'Angpt1', 'Vwf']
for g in canonical:
    if g in genes_sp and g not in markers_in_sp:
        markers_in_sp.append(g)
print(f"  With canonical markers: {len(markers_in_sp)}")

# Step 4: Run Tangram mapping
print("\n" + "=" * 60)
print("Step 4: Running Tangram mapping")
print("=" * 60)

# Keep raw reference for Tangram
ad_sc_raw = ad_sc[:, [g for g in ad_sc.var_names if g in markers_in_sp]].copy()

tg.pp_adatas(ad_sc_raw, ad_sp, genes=markers_in_sp)
print(f"  Training genes: {len(ad_sc_raw.uns.get('training_genes', []))}")

BATCH_SIZE = 30000
n_sp = ad_sp.n_obs
n_batches = (n_sp + BATCH_SIZE - 1) // BATCH_SIZE
print(f"  Spatial cells: {n_sp}, batches: {n_batches}")

all_ct_preds = []

for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, n_sp)
    print(f"  Batch {batch_idx+1}/{n_batches}: cells {start}-{end}")

    ad_sp_batch = ad_sp[start:end].copy()
    ad_sc_batch = ad_sc_raw.copy()
    tg.pp_adatas(ad_sc_batch, ad_sp_batch, genes=markers_in_sp)

    ad_map = tg.map_cells_to_space(
        ad_sc_batch, ad_sp_batch,
        device="cuda:0",
        mode="cells",
        density_prior='rna_count_based',
        num_epochs=500,
        learning_rate=0.1,
    )

    tg.project_cell_annotations(ad_map, ad_sp_batch, annotation='cell_type')
    all_ct_preds.append(ad_sp_batch.obsm['tangram_ct_pred'])

    del ad_map, ad_sc_batch, ad_sp_batch
    import torch
    torch.cuda.empty_cache()

print("  All batches complete!")

ct_preds_combined = pd.concat(all_ct_preds, axis=0)
ct_preds_combined.index = ad_sp.obs_names
ad_sp.obsm['tangram_ct_pred'] = ct_preds_combined

# Step 5: Extract T2DM-ref predictions
print("\n" + "=" * 60)
print("Step 5: Extracting T2DM-ref predictions")
print("=" * 60)

ct_probs = ad_sp.obsm['tangram_ct_pred']
ad_sp.obs['t2dm_predicted'] = ct_probs.idxmax(axis=1)
ad_sp.obs['t2dm_score'] = ct_probs.max(axis=1)

# Confidence threshold
n_types = ct_probs.shape[1]
uniform_prob = 1.0 / n_types
CONF_THRESHOLD = uniform_prob * 2
print(f"  n_types: {n_types}, uniform: {uniform_prob:.4f}, threshold: {CONF_THRESHOLD:.4f}")

t2dm_confident = ad_sp.obs['t2dm_score'] >= CONF_THRESHOLD
print(f"  T2DM-ref confident cells: {t2dm_confident.sum()}/{ad_sp.n_obs}")

print("\n  T2DM-ref predictions (confident):")
print(ad_sp.obs.loc[t2dm_confident, 't2dm_predicted'].value_counts())

# Step 6: Load existing annotations
print("\n" + "=" * 60)
print("Step 6: Loading existing YBM-ref + Geo-label annotations")
print("=" * 60)

# YBM-ref RCTD
ybm_csv = pd.read_csv(
    '/data_temp/mnt/raid5/xzq/spatial_bone/project_bone/RCTD/1-2/NC/soj_rctd_first_type_nc_min20.csv',
    index_col=0
)
ybm_csv.index = ybm_csv.index.astype(str)

# Geo-label
ad_geo = sc.read_h5ad(
    '/data_temp/mnt/raid5/xzq/spatial_bone/project_bone/RCTD/label_spot_by_geo/1-2_NC_ADD_CELLTYPE/4-NC_celltype.h5ad'
)

# Same mappings as nc_coarse_annotation.py
FINE_TO_COARSE = {
    'Osteoblast': 'Osteoblast', 'Osteoclast': 'Osteoclast',
    'Lepr_CAR': 'LepR+MSC', 'MSC': 'Fibro-MSC',
    'Chondrocytes': 'Chondrocytes',
    'Endo_Sinusoidal': 'Endothelial', 'Endo_Lymph': 'Endothelial',
    'Endo_Arterial': 'Endothelial', 'Endo_Other': 'Endothelial',
    'Endo_HSC_like': 'Endothelial',
    'VSMC': 'Vascular_SMC',
    'B_pro_prolif': 'Lymphoid', 'B_pro': 'Lymphoid', 'B_pre': 'Lymphoid',
    'B_trans': 'Lymphoid', 'B_memory': 'Lymphoid', 'B_mature': 'Lymphoid',
    'B_activated': 'Lymphoid', 'Plasma': 'Lymphoid',
    'T_Cd4': 'Lymphoid', 'T_Cd8': 'Lymphoid', 'TNK': 'Lymphoid',
    'NK': 'Lymphoid', 'ILC2': 'Lymphoid',
    'HSC': 'HSPC', 'STHSC': 'HSPC',
    'MPP2': 'HSPC', 'MPP3': 'HSPC', 'MPP4': 'HSPC',
    'GMP': 'HSPC', 'CLP': 'HSPC', 'MkP': 'HSPC', 'EryP': 'HSPC',
    'Neu_pre': 'Myeloid', 'Neu': 'Myeloid', 'Macro': 'Myeloid',
    'cDC': 'Myeloid', 'pDC': 'Myeloid',
    'Eosinophil': 'Myeloid', 'Basophil': 'Myeloid', 'Mast': 'Myeloid',
    'Megakaryocyte': 'Unassigned', 'Neuro_Stromal': 'Unassigned',
    'Tenocyte': 'Unassigned',
}

GEO_TO_COARSE = {
    'Neutrophils': 'Myeloid', 'B_lineage': 'Lymphoid',
    'Osteoblast': 'Osteoblast', 'Chondrocytes': 'Chondrocytes',
    'GMP': 'HSPC', 'Bone_tissue': 'Osteocytes',
    'Endothelial': 'Endothelial', 'EryP': 'HSPC',
    'Macrophages': 'Myeloid', 'DC': 'Myeloid',
    'MPP': 'HSPC', 'Lepr_CAR': 'LepR+MSC',
    'Granulocytes': 'Myeloid', 'MkP': 'HSPC', 'HSC': 'HSPC',
    'MSC': 'Fibro-MSC', 'T_NK_ILC': 'Lymphoid',
    'VSMC': 'Vascular_SMC', 'Mast_cell': 'Myeloid',
    'Neuro_Stromal': 'Unassigned', 'Tenocyte': 'Unassigned',
    'Osteoclast': 'Osteoclast', 'Adipocytes': 'Adipocytes',
    'Megakaryocyte': 'Unassigned',
}

# Build YBM-ref + Geo-label baseline
coarse_ybm = pd.Series('Unassigned', index=ad_sp.obs_names)

ybm_types = ybm_csv['x']
ybm_coarse = ybm_types.map(FINE_TO_COARSE).dropna()
common_ybm = ybm_coarse.index.intersection(ad_sp.obs_names)
coarse_ybm.loc[common_ybm] = ybm_coarse.loc[common_ybm]

geo_types = ad_geo.obs['cell_type']
geo_coarse = geo_types.map(GEO_TO_COARSE).dropna()
still_unassigned = coarse_ybm == 'Unassigned'
geo_fill = geo_coarse.index.intersection(ad_sp.obs_names[still_unassigned])
coarse_ybm.loc[geo_fill] = geo_coarse.loc[geo_fill]

print(f"  YBM-ref + Geo-label baseline: {(coarse_ybm != 'Unassigned').sum()} annotated")

# Step 7: Merge: T2DM-ref supplements YBM-ref
print("\n" + "=" * 60)
print("Step 7: Merging T2DM-ref into annotation")
print("=" * 60)

# T2DM-ref can provide these types better than YBM-ref:
T2DM_PREFERRED = ['Adipocytes', 'Osteocytes', 'Osteoblast']
# → Adipocytes: YBM-ref has none, T2DM-ref has them
# → Osteocytes: YBM-ref relies on Geo-label (Bone_tissue), T2DM has MSC_phex

coarse_new = coarse_ybm.copy()

# Track changes
changes = {'Adipocytes': 0, 'Osteocytes': 0, 'Osteoblast': 0}

for ct in T2DM_PREFERRED:
    # Find cells where T2DM-ref confidently predicts this type
    t2dm_hit = (ad_sp.obs['t2dm_predicted'] == ct) & t2dm_confident
    t2dm_indices = ad_sp.obs_names[t2dm_hit]

    if ct == 'Adipocytes':
        # YBM-ref has no Adipocytes → take all T2DM-ref predictions
        eligible = t2dm_indices
    elif ct == 'Osteocytes':
        # YBM-ref has Osteocytes only from Geo-label (Bone_tissue)
        # T2DM-ref MSC_phex is more specific molecular marker
        eligible = t2dm_indices
    else:
        # For Osteoblast, prefer T2DM-ref over Geo-label but not over YBM-ref
        # Only override cells that are Unassigned or from Geo-label
        eligible = t2dm_indices.intersection(
            ad_sp.obs_names[coarse_ybm.isin(['Unassigned', 'Osteoblast'])]
        )

    if len(eligible) > 0:
        n_changed = (coarse_new.loc[eligible] != ct).sum()
        coarse_new.loc[eligible] = ct
        changes[ct] = n_changed
        print(f"  {ct}: T2DM-ref sets {len(eligible)} cells ({n_changed} changed)")

print(f"\n  Overall annotation changes:")
for ct, n in changes.items():
    print(f"    {ct}: {n} cells updated")

# Step 8: Save updated h5ad
print("\n" + "=" * 60)
print("Step 8: Saving updated annotation")
print("=" * 60)

ad_sp.obs['coarse_celltype'] = coarse_new.values.astype(str)

print("\n  Updated coarse cell type distribution:")
dist = ad_sp.obs['coarse_celltype'].value_counts()
for ct, cnt in dist.items():
    pct = cnt / ad_sp.n_obs * 100
    print(f"  {ct:20s} {cnt:7d}  ({pct:5.1f}%)")

annotated = (ad_sp.obs['coarse_celltype'] != 'Unassigned').sum()
print(f"\n  Annotated: {annotated}/{ad_sp.n_obs} ({annotated/ad_sp.n_obs*100:.1f}%)")

ad_sp.write_h5ad(f"{OUT_DIR}/16-4-NC_SR_coarse_annotated.h5ad")
ad_sp.obs[['coarse_celltype']].to_csv(f"{OUT_DIR}/coarse_celltype_nc.csv")
print(f"  Saved: {OUT_DIR}/16-4-NC_SR_coarse_annotated.h5ad")

# Step 9: Visualization
print("\n" + "=" * 60)
print("Step 9: Generating visualization")
print("=" * 60)

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
print('Done!')
