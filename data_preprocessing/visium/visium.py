import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json2
from utils import getGradientMagnitude, get_matrix_from_h5, remove_all_file
from skimage.measure import shannon_entropy
import scipy.sparse as sp_sparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ----------------------------------------
# Configuration
# ----------------------------------------
super_root = r'/date/qnf/VisiumHD/visium/Visium_polya_mousebrain/'
root = r'/date/qnf/VisiumHD/visium/Visium_polya_mousebrain'
ori_HE_name = 'Visium_Fresh_Frozen_Adult_Mouse_Brain_image.tif'
out_fileName = '20221018'
crop_mpp = 0.5
crop_pixel = 5120
sub_crop_pixel = 320
resize_time = 20
Gene_num = 32285
feature_matrix = 'Visium_Fresh_Frozen_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5'

# ----------------------------------------
# Load data once
# ----------------------------------------
img_array = np.array(Image.open(os.path.join(root, ori_HE_name)))
json_data = json2.load_file(os.path.join(root, 'spatial/scalefactors_json.json'))
MPP = 65.0 / json_data['spot_diameter_fullres']

# Pre-resize for WSI
wsi_w_MPP05 = int(img_array.shape[1] * (MPP / crop_mpp))
wsi_h_MPP05 = int(img_array.shape[0] * (MPP / crop_mpp))
im_resized = cv2.resize(img_array, (wsi_w_MPP05, wsi_h_MPP05), interpolation=cv2.INTER_LINEAR)
im = np.transpose(im_resized, (1, 0, 2))  # [width, height, 3]

# Load position info
posi_csv = 'tissue_positions.csv' if out_fileName == '20221018' else 'tissue_positions_list.csv'
header = None if out_fileName == '202210181' else 0
posi_info = pd.read_csv(os.path.join(root, 'spatial', posi_csv), header=header).values

# Load ST matrix
filtered_matrix_h5 = get_matrix_from_h5(os.path.join(root, feature_matrix))
barcodes = filtered_matrix_h5.barcodes
feature_ref = filtered_matrix_h5.feature_ref
matrix = filtered_matrix_h5.matrix[:Gene_num, :]
feature_ref['name'] = feature_ref['name'][:Gene_num]

# Prepare gene groups
csv_gene_groups = os.path.join(super_root, 'gene_groups.csv')
if not os.path.exists(csv_gene_groups):
    gene_names = [n.decode('utf-8') for n in feature_ref['name']]
    pd.DataFrame({'gene_names': gene_names}).to_csv(csv_gene_groups, index=False)
gene_groups_all = [bytes(n, 'utf-8') for n in pd.read_csv(csv_gene_groups).values[:, 0]]

# ----------------------------------------
# Parallel functions
# ----------------------------------------
def process_wsi_patch(idx_pair):
    i, j = idx_pair
    count_spot = 0
    for spot in range(posi_info.shape[0]):
        h_spot = posi_info[spot, 4] * (MPP / crop_mpp)
        w_spot = posi_info[spot, 5] * (MPP / crop_mpp)
        if h_spot < im.shape[1] and w_spot < im.shape[0] and posi_info[spot, 1]:
            if i * crop_pixel/2 <= w_spot < i * crop_pixel/2 + crop_pixel and \
               j * crop_pixel/2 <= h_spot < j * crop_pixel/2 + crop_pixel:
                count_spot += 1
    if count_spot <= 180:
        return

    out_wsi = os.path.join(super_root, 'WSI/extract', out_fileName, f"{i}_{j}")
    for d in (out_wsi,):
        if not os.path.exists(d): os.makedirs(d)
        else: remove_all_file(d)

    x0, y0 = int(i * crop_pixel/2), int(j * crop_pixel/2)
    patch = im[x0:x0+crop_pixel, y0:y0+crop_pixel, :]

    collected = []
    for w in range(crop_pixel // sub_crop_pixel):
        for h in range(crop_pixel // sub_crop_pixel):
            sub = patch[w*sub_crop_pixel:(w+1)*sub_crop_pixel,
                       h*sub_crop_pixel:(h+1)*sub_crop_pixel, :]
            sub_t = np.transpose(sub, (1,0,2))
            r_avg = np.mean(sub_t[...,0])
            bright = np.mean(0.2126*sub_t[...,0] + 0.7152*sub_t[...,1] + 0.0722*sub_t[...,2])
            rgb = Image.fromarray(sub_t.astype('uint8')).convert('RGB')
            grad = getGradientMagnitude(np.array(rgb))
            unique, counts = np.unique(grad, return_counts=True)
            if counts[np.where(unique <= 20)].sum() < sub_crop_pixel**2*0.75:
                if bright < 220 and r_avg > 120:
                    collected.append(np.array(rgb))
    if not collected: return
    smalls = [np.asarray(Image.fromarray(img).resize(
               (sub_crop_pixel//resize_time, sub_crop_pixel//resize_time)))
              for img in collected]
    total = (sub_crop_pixel//resize_time)**2
    reps, rem = divmod(total, len(smalls))
    tiled = smalls * reps + smalls[:rem]
    ent = np.array([shannon_entropy(img) for img in tiled])
    order = np.argsort(ent)
    np.save(os.path.join(out_wsi, '320_to16.npy'), np.array(tiled)[order])

    patch_tr = np.transpose(patch, (1,0,2))
    resized = cv2.resize(patch_tr, (crop_pixel//resize_time, crop_pixel//resize_time),
                         interpolation=cv2.INTER_LINEAR)[...,:3]
    np.save(os.path.join(out_wsi, '5120_to256.npy'), resized)


def process_spot_patch(idx_pair):
    i, j = idx_pair
    count_spot = 0
    for spot in range(posi_info.shape[0]):
        h_spot = posi_info[spot, 4] * (MPP / crop_mpp)
        w_spot = posi_info[spot, 5] * (MPP / crop_mpp)
        if h_spot < wsi_h_MPP05 and w_spot < wsi_w_MPP05 and posi_info[spot, 1]:
            if i*2560 <= w_spot < i*2560+5120 and j*2560 <= h_spot < j*2560+5120:
                count_spot += 1
    if count_spot <= 180:
        return

    out_spot = os.path.join(super_root, 'spot_ST/extract', out_fileName, f"{i}_{j}")
    if not os.path.exists(out_spot): os.makedirs(out_spot)
    else: remove_all_file(out_spot)

    grid = np.zeros((26,26,len(gene_groups_all)), dtype=np.int32)
    for g_idx, g in enumerate(gene_groups_all):
        if g in feature_ref['name']:
            feat = np.where(feature_ref['name']==g)[0][0]
            for spot in range(posi_info.shape[0]):
                h_spot = posi_info[spot, 4] * (MPP / crop_mpp)
                w_spot = posi_info[spot, 5] * (MPP / crop_mpp)
                if i*2560 <= w_spot < i*2560+5120 and j*2560 <= h_spot < j*2560+5120 and posi_info[spot,1]:
                    x = int((w_spot - i*2560) / 200)
                    y = int((h_spot - j*2560) / 200)
                    bc = bytes(posi_info[spot,0], 'UTF-8')
                    grid[y, x, g_idx] = matrix[feat, np.where(barcodes==bc)[0][0]]
    shp = grid.shape
    mat2d = grid.reshape(-1, shp[2])
    sp = sp_sparse.csr_matrix(mat2d)
    sp_sparse.save_npz(os.path.join(out_spot, 'spot_ST.npz'), sp)


def main():
    # WSI patches
    max_i = int(im.shape[0] / (crop_pixel/2) - 1)
    max_j = int(im.shape[1] / (crop_pixel/2) - 1)
    wsi_tasks = [(i, j) for i in range(max_i) for j in range(max_j)]
    with Pool(processes=min(cpu_count(), len(wsi_tasks))) as pool:
        for _ in tqdm(pool.imap_unordered(process_wsi_patch, wsi_tasks),
                       total=len(wsi_tasks), desc='Processing WSI'):
            pass

    # Spot-ST patches
    max_i2 = int(wsi_w_MPP05 / 2560 - 1)
    max_j2 = int(wsi_h_MPP05 / 2560 - 1)
    spot_tasks = [(i, j) for i in range(max_i2) for j in range(max_j2)]
    with Pool(processes=min(cpu_count(), len(spot_tasks))) as pool:
        for _ in tqdm(pool.imap_unordered(process_spot_patch, spot_tasks),
                       total=len(spot_tasks), desc='Processing Spot‑ST'):
            pass

if __name__ == '__main__':
    main()