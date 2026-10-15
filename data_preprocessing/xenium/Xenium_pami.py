#%%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pathlib
from utils import *
from utils import warpAffine_new
from skimage.measure import shannon_entropy
from pyometiff import OMETIFFReader
import gc
def _invertAffineTransform(matrix):

    matrix = np.concatenate([matrix, np.array([0, 0, 1], dtype=matrix.dtype)[None]])
    return inv(matrix)[:2]
###################################
######## WSI
###################################
crop_mpp=0.5
crop_pixel=5120
sub_crop_pixel=320
resize_time=20

root=r'/home/cbtil4/ST/datasets/output-XETG00166__0041204__TMA1__20241031__051835/'
save_HE_root=r'/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/WSI/extract/20241031/'
root=r'/home/cbtil4/ST/datasets/'
save_HE_root=r'/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/'
ori_HE_name='TMA1_40X_ZLIB_2_1024.ome.tif'
alignment_file='alignment.csv'


# if not os.path.exists(root+'/output-XETG00166__0041204__TMA1__20241031__051835/morphology_img_info.npy'):
#     img_fpath = pathlib.Path(root+'/output-XETG00166__0041204__TMA1__20241031__051835/morphology.ome.tif')
#     reader = OMETIFFReader(fpath=img_fpath)
#     img_array, metadata, xml_metadata = reader.read()
#     if img_array.shape[0]==3:
#         img_array = np.transpose(img_array, axes=(1, 2, 0))
#     if img_array.shape[0]>1000:
#         wsi_h=img_array.shape[0]
#         wsi_w=img_array.shape[1]
#     else:
#         wsi_h = img_array.shape[1]
#         wsi_w = img_array.shape[2]
#     morphology_img_info=np.array([0,0,0.0])
#     morphology_img_info[0]=wsi_h
#     morphology_img_info[1]=wsi_w
#     f = open(root+'/output-XETG00166__0041204__TMA1__20241031__051835/experiment.xenium')
#     data = json.load(f)
#     morphology_img_info[2]=data['pixel_size']
#     np.save(root+'/output-XETG00166__0041204__TMA1__20241031__051835/morphology_img_info.npy',morphology_img_info)




morphology_img_info=np.load(root+'/output-XETG00166__0041204__TMA1__20241031__051835/morphology_img_info.npy')
wsi_h_MPP05=int(morphology_img_info[0]*(morphology_img_info[2]/crop_mpp))
wsi_w_MPP05=int(morphology_img_info[1]*(morphology_img_info[2]/crop_mpp))
img_fpath = pathlib.Path(root + '/output-XETG00166__0041204__TMA1__20241031__051835/' + ori_HE_name)
reader = OMETIFFReader(fpath=img_fpath)
img_array, metadata, xml_metadata = reader.read()
if img_array.shape[0]==3:
    img_array = np.transpose(img_array, axes=(1, 2, 0))

### ONLY FOR THOESE NEED ALIGNMENT
if 1:
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    alignment = pd.read_csv(root +'output-XETG00166__0041204__TMA1__20241031__051835/'+ alignment_file, header=None).values[0:2, :]
    if img_array.shape[0]< 32766 and img_array.shape[1]<32766:
        img_array = cv2.warpAffine(img_array,alignment,(int(morphology_img_info[1]),int(morphology_img_info[0])))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        img_array = warpAffine_new(img_array, alignment, (int(morphology_img_info[1]),int(morphology_img_info[0])))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(img_array, (0, 0, 0), (0, 0, 0))
        mask_inv = cv2.bitwise_not(mask)
        x, y, w, h = cv2.boundingRect(mask_inv)
        img_array = img_array[y:y + h, x:x + w]

img_array = cv2.resize(img_array, (wsi_w_MPP05, wsi_h_MPP05), interpolation=cv2.INTER_LINEAR)
im=np.transpose(img_array,(1,0,2))
im=plt.imread(root+'aligned_HE.tif', format="tiff")


img_w=im.shape[0]
img_h=im.shape[1]

#%%

count_img=0
for i in range(np.int64(img_w / int(crop_pixel/2)-1)):
    for j in range(np.int64(img_h / int(crop_pixel/2)-1)):
        if not os.path.exists(save_HE_root+ '/'+str(i) + '_' + str(j)):
            os.makedirs(save_HE_root+ '/'+str(i) + '_' + str(j))
        else:
            remove_all_file(save_HE_root+ '/'+str(i) + '_' + str(j))

        rgba_image_5120 = im[int(i*crop_pixel/2):int(i*crop_pixel/2+crop_pixel),int(j*crop_pixel/2):int(j*crop_pixel/2+crop_pixel),:]
        count = 0

        for w in range(int(crop_pixel/sub_crop_pixel)):
            for h in range(int(crop_pixel/sub_crop_pixel)):
                rgba_image_320 = rgba_image_5120[w * sub_crop_pixel:(w + 1) * sub_crop_pixel, h * sub_crop_pixel:(h + 1) * sub_crop_pixel, :]
                rgba_image_pil = np.transpose(rgba_image_320, (1, 0, 2))
                r_ = rgba_image_pil[..., 0]
                g_ = rgba_image_pil[..., 1]
                b_ = rgba_image_pil[..., 2]
                r_avg = np.mean(r_)
                bright_avg = np.mean(0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_)
                rgba_image_pil = Image.fromarray(rgba_image_pil.astype('uint8'))
                im0 = rgba_image_pil.convert("RGB")
                grad = getGradientMagnitude(np.array(im0))
                unique, counts = np.unique(grad, return_counts=True)
                if counts[np.argwhere(unique <= 20)].sum() < sub_crop_pixel * sub_crop_pixel * 0.75:
                    if bright_avg < 220 and r_avg > 120:
                        im0 = np.array(im0)
                        im0 = np.expand_dims(im0, 0)
                        if count == 0:
                            rgba_image_320_save = im0
                        else:
                            rgba_image_320_save = np.concatenate((rgba_image_320_save, im0), axis=0)
                        count += 1

        ### generate WSI_320_to_16
        WSI_320_to16=[]
        WSI_320=rgba_image_320_save
        for k in range(WSI_320.shape[0]):
            im11= Image.fromarray(WSI_320[k])
            im11 = im11.resize((int(sub_crop_pixel/resize_time), int(sub_crop_pixel/resize_time)))
            im11 = np.asarray(im11)
            WSI_320_to16.append(im11)

        WSI_320_to16_NEW=WSI_320_to16
        times = int(np.floor(int(sub_crop_pixel/resize_time)*int(sub_crop_pixel/resize_time) / WSI_320.shape[0]))
        remaining = int(sub_crop_pixel/resize_time)*int(sub_crop_pixel/resize_time) % WSI_320.shape[0]
        if times > 1:
            for k in range(times - 1):
                WSI_320_to16_NEW = WSI_320_to16_NEW + WSI_320_to16
        if not remaining == 0:
            WSI_320_to16_NEW = WSI_320_to16_NEW + WSI_320_to16[0:remaining]
        WSI_320_to16_NEW = np.array(WSI_320_to16_NEW)
        entropy=[]
        for ii in range(WSI_320_to16_NEW.shape[0]):
            entropy.append(np.round(shannon_entropy(WSI_320_to16_NEW[ii])))
        entropy=np.array(entropy)
        entropy_order=np.argsort(entropy)
        WSI_320_to16_entropy=WSI_320_to16_NEW[entropy_order]
        np.save(save_HE_root+ '/'+str(i) + '_' + str(j) + '/320_to16.npy',WSI_320_to16_entropy)
        rgba_image_5120 = np.transpose(rgba_image_5120, (1, 0, 2))
        rgba_image_5120 = cv2.resize(rgba_image_5120, (int(crop_pixel), int(crop_pixel)), interpolation=cv2.INTER_LINEAR)[...,0:3]
        np.save(save_HE_root+ '/'+str(i) + '_' + str(j) + '/5120.npy',rgba_image_5120)
        rgba_image_5120 = np.transpose(rgba_image_5120, (1, 0, 2))
        rgba_image_5120 = cv2.resize(rgba_image_5120, (int(crop_pixel/resize_time), int(crop_pixel/resize_time)), interpolation=cv2.INTER_LINEAR)[...,0:3]
        np.save(save_HE_root+ '/'+str(i) + '_' + str(j) + '/5120_to256.npy',rgba_image_5120)
        print(i, j)
a=1
# ###################################
# ######## HR ST and spot_ST
# ###################################

crop_wsi_mpp=0.5
st_rela_wsi_mpp=20
spotst_rela_wsi_mpp=200
crop_wsi_pixel=5120
#
#
gene_groups='/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/gene_groups.csv'
root=r'/home/cbtil4/ST/datasets/output-XETG00166__0041204__TMA1__20241031__051835/'
save_HRST_root=r'/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/HR_ST/extract/20241031/'
save_spotST_root=r'/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/spot_ST/extract/20241031/'
save_WSI_root=r'/home/cbtil4/ST/datasets/Xenium5k_Craniopharyngioma/WSI/extract/20241031/'
#
#
#
# import pyarrow.parquet as pq
# morphology_img_info=np.load(root+'/morphology_img_info.npy')
# wsi_h_MPP05=int(morphology_img_info[0]*(morphology_img_info[2]/crop_wsi_mpp))
# wsi_w_MPP05=int(morphology_img_info[1]*(morphology_img_info[2]/crop_wsi_mpp))
# img_w=wsi_w_MPP05
# img_h= wsi_h_MPP05
# gene_groups_all=pd.read_csv(gene_groups).values[:,0]
# gene_groups_all_str=list(gene_groups_all)
# for i in range(gene_groups_all.shape[0]):
#     gene_groups_all[i]=bytes(gene_groups_all[i],'UTF-8')
# gene_groups_all_bytes=list(gene_groups_all)
# print('1000')
# transcripts = pd.read_parquet(root+'/transcripts.parquet')
# if isinstance(transcripts.values[0, 3], str):
#     gene_groups_all = gene_groups_all_str
# else:
#     gene_groups_all = gene_groups_all_bytes
# print("read transcripts finished.")
# count=0



# for i in range(np.int64(img_w / int(crop_wsi_pixel/2)-1)):
#     for j in range(np.int64(img_h / int(crop_wsi_pixel/2)-1)):
#         if os.path.isdir(save_WSI_root+str(i) + '_' + str(j)):
#             if not os.path.exists(save_HRST_root+ '/'+str(i) + '_' + str(j)):
#                 os.makedirs(save_HRST_root+ '/'+str(i) + '_' + str(j))
#             else:
#                 remove_all_file(save_HRST_root+ '/'+str(i) + '_' + str(j))
#             if not os.path.exists(save_spotST_root+ '/'+str(i) + '_' + str(j)):
#                 os.makedirs(save_spotST_root+ '/'+str(i) + '_' + str(j))
#             else:
#                 remove_all_file(save_spotST_root+ '/'+str(i) + '_' + str(j))


#             gene_save256 = np.zeros(shape=(int(crop_wsi_pixel/st_rela_wsi_mpp),int(crop_wsi_pixel/st_rela_wsi_mpp),len(gene_groups_all)),dtype=np.int16)
#             gene_save128 = np.zeros(shape=(int(crop_wsi_pixel/st_rela_wsi_mpp/2),int(crop_wsi_pixel/st_rela_wsi_mpp/2),len(gene_groups_all)),dtype=np.int16)
#             gene_saveSPOT = np.zeros(shape=(int(crop_wsi_pixel/spotst_rela_wsi_mpp)+1,int(crop_wsi_pixel/spotst_rela_wsi_mpp)+1,len(gene_groups_all)),dtype=np.int16)


#             transcripts_temp = transcripts.loc[transcripts['x_location'] >= 0.5*i * (crop_wsi_pixel * crop_wsi_mpp)]
#             transcripts_temp = transcripts_temp.loc[transcripts_temp['x_location'] < (0.5*i + 1) * (crop_wsi_pixel * crop_wsi_mpp)]
#             transcripts_temp = transcripts_temp.loc[transcripts_temp['y_location'] >= 0.5*j * (crop_wsi_pixel * crop_wsi_mpp)]
#             transcripts_temp = transcripts_temp.loc[transcripts_temp['y_location'] < (0.5*j + 1) * (crop_wsi_pixel * crop_wsi_mpp)].values
#             print("1111")
#             if transcripts_temp.shape[0] :
#                 for transcript in range(transcripts_temp.shape[0]):
#                     if transcripts_temp[transcript, 3] in gene_groups_all:
#                         gene_index = gene_groups_all.index(transcripts_temp[transcript, 3])
#                         w = transcripts_temp[transcript, 4] -(0.5*i) * (crop_wsi_pixel * crop_wsi_mpp)
#                         h = transcripts_temp[transcript, 5] -(0.5*j) * (crop_wsi_pixel * crop_wsi_mpp)
#                         w_HRST = int(w / crop_wsi_mpp/st_rela_wsi_mpp)
#                         h_HRST = int(h / crop_wsi_mpp/st_rela_wsi_mpp)
#                         gene_save256[h_HRST, w_HRST, gene_index] += 1
#                         w_HRST128 = int(w / crop_wsi_mpp / st_rela_wsi_mpp/2)
#                         h_HRST128 = int(h / crop_wsi_mpp / st_rela_wsi_mpp/2)
#                         gene_save128[h_HRST128, w_HRST128, gene_index] += 1
#                         w_HRST_spot = int(w / crop_wsi_mpp / spotst_rela_wsi_mpp)
#                         h_HRST_spot = int(h / crop_wsi_mpp / spotst_rela_wsi_mpp)
#                         gene_saveSPOT[h_HRST_spot, w_HRST_spot, gene_index] += 1
#                         print(i, j, transcript / transcripts_temp.shape[0])

#                 shape=gene_save256.shape
#                 gene_save256_2d=gene_save256.reshape(-1,shape[2])
#                 sparse_matrix = sp_sparse.csr_matrix(gene_save256_2d)
#                 sp_sparse.save_npz(save_HRST_root+ '/'+str(i) + '_' + str(j) + '/' + 'HR_ST_256.npz', sparse_matrix)

#                 shape = gene_save128.shape
#                 gene_save128_2d = gene_save128.reshape(-1, shape[2])
#                 sparse_matrix = sp_sparse.csr_matrix(gene_save128_2d)
#                 sp_sparse.save_npz(save_HRST_root + '/' + str(i) + '_' + str(j) + '/' + 'HR_ST_128.npz', sparse_matrix)

#                 shape = gene_saveSPOT.shape
#                 gene_saveSPOT_2d = gene_saveSPOT.reshape(-1, shape[2])
#                 sparse_matrix = sp_sparse.csr_matrix(gene_saveSPOT_2d)
#                 sp_sparse.save_npz(save_spotST_root + '/' + str(i) + '_' + str(j) + '/' + 'spot_ST.npz', sparse_matrix)
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import duckdb
import scipy.sparse as sp_sparse

def remove_all_file(dir_path: str):
    if not os.path.exists(dir_path):
        return
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.unlink(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except Exception as e:
            print(f"[warn] 删除失败: {p} -> {e}")


def detect_columns_with_duckdb(con, parquet_path: str):

    parquet_sql = parquet_path.replace("'", "''")  # 安全转义
    cols_df = con.execute(
        f"SELECT * FROM read_parquet('{parquet_sql}') LIMIT 0"
    ).df()
    return list(cols_df.columns)


def choose_gene_column(columns):

    lower_map = {c.lower(): c for c in columns}
    candidates = ["gene_symbol", "gene", "feature_name", "geneid", "symbol"]
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
  


def build_sparse_accum(rows, cols, chans, shape_rc, n_channels, dtype=np.int16):
 
    flat_row = rows * shape_rc[1] + cols  # row = h*W + w
    data = np.ones(len(chans), dtype=dtype)
    coo = sp_sparse.coo_matrix(
        (data, (flat_row, chans)),
        shape=(shape_rc[0] * shape_rc[1], n_channels),
        dtype=dtype
    )
    return coo.tocsr()  # 合并重复项


def main():
    PARQUET_PATH = os.path.join(root, "transcripts.parquet")
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"找不到 parquet 文件: {PARQUET_PATH}")

    # ---------- 读 morphology 信息，计算 img_w/img_h ----------
    morphology = np.load(os.path.join(root, "morphology_img_info.npy"))
    wsi_h_MPP05 = int(morphology[0] * (morphology[2] / crop_wsi_mpp))
    wsi_w_MPP05 = int(morphology[1] * (morphology[2] / crop_wsi_mpp))
    img_w, img_h = wsi_w_MPP05, wsi_h_MPP05

    # ---------- 读基因清单，统一成字符串，构造映射 ----------
    gene_series = pd.read_csv(gene_groups).iloc[:, 0].astype("string")
    gene_series = gene_series.dropna().astype(str)  # 去除 NA，统一 str
    gene_to_idx = {g: i for i, g in enumerate(gene_series.tolist())}
    gene_set = set(gene_to_idx.keys())
    C = len(gene_to_idx)
    print(f"[info] 基因通道数: {C}")

    # ---------- DuckDB 连接 & 注册基因维表 ----------
    con = duckdb.connect()
    con.register("genes_df", pd.DataFrame({"gene_symbol": gene_series}))

    # ---------- 自动识别列名 ----------
    columns = detect_columns_with_duckdb(con, PARQUET_PATH)
    lower_map = {c.lower(): c for c in columns}

    # 坐标列（按需修改别名）
    X_COL = lower_map.get("x_location", None)
    Y_COL = lower_map.get("y_location", None)
 
    # 基因列
    GENE_COL = choose_gene_column(columns)
    print(f"[info] 识别到列: X={X_COL}, Y={Y_COL}, GENE={GENE_COL}")

    # ---------- 预计算各尺度栅格大小 ----------
    H256 = int(crop_wsi_pixel / st_rela_wsi_mpp)
    W256 = int(crop_wsi_pixel / st_rela_wsi_mpp)
    H128 = int(crop_wsi_pixel / (st_rela_wsi_mpp * 2))
    W128 = int(crop_wsi_pixel / (st_rela_wsi_mpp * 2))
    HSP  = int(crop_wsi_pixel /  spotst_rela_wsi_mpp) + 1
    WSP  = int(crop_wsi_pixel /  spotst_rela_wsi_mpp) + 1

    win_size_phy = crop_wsi_pixel * crop_wsi_mpp
    step = int(crop_wsi_pixel / 2)

    parquet_sql = PARQUET_PATH.replace("'", "''")  # SQL 字面量安全转义

    # ---------- 主循环（与原逻辑等价） ----------
    n_i = int(img_w / step - 1)
    n_j = int(img_h / step - 1)
    total_tiles = n_i * n_j
    done = 0

    for i in range(n_i):
        for j in range(n_j):
            print(i,j)
            # 你的脚本里：只有当 save_WSI_root/{i}_{j} 目录存在时才处理
            # tile_dir_wsi = os.path.join(save_WSI_root, f"{i}_{j}")
            # if not os.path.isdir(tile_dir_wsi):
            #     continue

            tile_hrst_dir = os.path.join(save_HRST_root, f"{i}_{j}")
            tile_spot_dir = os.path.join(save_spotST_root, f"{i}_{j}")
            os.makedirs(tile_hrst_dir, exist_ok=True)
            os.makedirs(tile_spot_dir, exist_ok=True)
            remove_all_file(tile_hrst_dir)
            remove_all_file(tile_spot_dir)

            # 与你原公式一致的窗口边界（注意 0.5*i 偏移）
            x0 = 0.5 * i * win_size_phy
            x1 = (0.5 * i + 1) * win_size_phy
            y0 = 0.5 * j * win_size_phy
            y1 = (0.5 * j + 1) * win_size_phy

            # 仅读取该窗口命中的行，并统一 gene 为字符串；再与 genes_df 内连接
            q = f"""
            WITH t AS (
              SELECT
                {X_COL} AS x_location,
                {Y_COL} AS y_location,
                CAST({GENE_COL} AS VARCHAR) AS gene_symbol
              FROM read_parquet('{parquet_sql}')
              WHERE {X_COL} >= {x0} AND {X_COL} < {x1}
                AND {Y_COL} >= {y0} AND {Y_COL} < {y1}
            )
            SELECT x_location, y_location, t.gene_symbol
            FROM t
            JOIN genes_df g
              ON t.gene_symbol = g.gene_symbol
            """
            df = con.execute(q).df()
            if df.empty:
                done += 1
                if done % 50 == 0:
                    print(f"[progress] {done}/{total_tiles} tiles processed...")
                continue

            # ========== 向量化坐标到各尺度网格 ==========
            w = df["x_location"].to_numpy(dtype=np.float64) - x0
            h = df["y_location"].to_numpy(dtype=np.float64) - y0

            w_HRST    = (w / crop_wsi_mpp /  st_rela_wsi_mpp).astype(np.int64)
            h_HRST    = (h / crop_wsi_mpp /  st_rela_wsi_mpp).astype(np.int64)
            w_HRST128 = (w / crop_wsi_mpp / (st_rela_wsi_mpp * 2)).astype(np.int64)
            h_HRST128 = (h / crop_wsi_mpp / (st_rela_wsi_mpp * 2)).astype(np.int64)
            w_SPOT    = (w / crop_wsi_mpp /  spotst_rela_wsi_mpp).astype(np.int64)
            h_SPOT    = (h / crop_wsi_mpp /  spotst_rela_wsi_mpp).astype(np.int64)

            # 边界保护，避免越界
            np.clip(w_HRST,    0, W256-1, out=w_HRST)
            np.clip(h_HRST,    0, H256-1, out=h_HRST)
            np.clip(w_HRST128, 0, W128-1, out=w_HRST128)
            np.clip(h_HRST128, 0, H128-1, out=h_HRST128)
            np.clip(w_SPOT,    0, WSP-1,  out=w_SPOT)
            np.clip(h_SPOT,    0, HSP-1,  out=h_SPOT)

            # 基因 -> 通道索引（向量化映射）
            gene_idx = df["gene_symbol"].map(gene_to_idx).to_numpy()
            # （理论上不应有 -1；若 gene_groups 与数据不一致，可在此处过滤 gene_idx==-1）

            # ========== 稀疏累加并保存 ==========
            mat256 = build_sparse_accum(h_HRST,    w_HRST,    gene_idx, (H256, W256), C, dtype=np.int16)
            mat128 = build_sparse_accum(h_HRST128, w_HRST128, gene_idx, (H128, W128), C, dtype=np.int16)
            matSP  = build_sparse_accum(h_SPOT,    w_SPOT,    gene_idx, (HSP,  WSP ), C, dtype=np.int16)

            sp_sparse.save_npz(os.path.join(tile_hrst_dir, "HR_ST_256.npz"), mat256)
            sp_sparse.save_npz(os.path.join(tile_hrst_dir, "HR_ST_128.npz"), mat128)
            sp_sparse.save_npz(os.path.join(tile_spot_dir, "spot_ST.npz"),   matSP)

            done += 1
            if done % 50 == 0:
                print(f"[progress] {done}/{total_tiles} tiles processed...")

    print("[done] 全部 tile 处理完成。")


if __name__ == "__main__":
    main()



# for i in range(np.int64(img_w / (crop_wsi_pixel / 2) - 1)):
#     for j in range(np.int64(img_h / (crop_wsi_pixel / 2) - 1)):
#         block_dir_hrst = os.path.join(save_HRST_root, f'{i}_{j}')
#         block_dir_spot = os.path.join(save_spotST_root, f'{i}_{j}')

#         os.makedirs(block_dir_hrst, exist_ok=True)
#         os.makedirs(block_dir_spot, exist_ok=True)

#         # 只初始化当前块所需的大小
#         gene_save256 = np.zeros(shape=(int(crop_wsi_pixel / st_rela_wsi_mpp),
#                                        int(crop_wsi_pixel / st_rela_wsi_mpp), len(gene_groups_all)), 
#                                 dtype=np.int16)

#         gene_save128 = np.zeros(shape=(int(crop_wsi_pixel / st_rela_wsi_mpp / 2),
#                                        int(crop_wsi_pixel / st_rela_wsi_mpp / 2), len(gene_groups_all)), 
#                                 dtype=np.int16)

#         gene_saveSPOT = np.zeros(shape=(int(crop_wsi_pixel / spotst_rela_wsi_mpp) + 1,
#                                          int(crop_wsi_pixel / spotst_rela_wsi_mpp) + 1, len(gene_groups_all)), 
#                                   dtype=np.int16)

#         # 获取当前块的转录本信息
#         transcripts_temp = transcripts.loc[(transcripts['x_location'] >= 0.5 * i * (crop_wsi_pixel * crop_wsi_mpp)) & 
#                                            (transcripts['x_location'] < (0.5 * i + 1) * (crop_wsi_pixel * crop_wsi_mpp)) & 
#                                            (transcripts['y_location'] >= 0.5 * j * (crop_wsi_pixel * crop_wsi_mpp)) & 
#                                            (transcripts['y_location'] < (0.5 * j + 1) * (crop_wsi_pixel * crop_wsi_mpp))].values

#         # 若transcripts_temp不为空，则进行处理
#         if transcripts_temp.shape[0] > 0:
#             for transcript in range(transcripts_temp.shape[0]):
#                 if transcripts_temp[transcript, 3] in gene_groups_all:
#                     gene_index = gene_groups_all.index(transcripts_temp[transcript, 3])
                    
#                     w = transcripts_temp[transcript, 4] - (0.5 * i) * (crop_wsi_pixel * crop_wsi_mpp)
#                     h = transcripts_temp[transcript, 5] - (0.5 * j) * (crop_wsi_pixel * crop_wsi_mpp)

#                     # 更新HRST 256
#                     w_HRST = int(w / crop_wsi_mpp / st_rela_wsi_mpp)
#                     h_HRST = int(h / crop_wsi_mpp / st_rela_wsi_mpp)
#                     gene_save256[h_HRST, w_HRST, gene_index] += 1

#                     # 更新HRST 128
#                     w_HRST128 = int(w / crop_wsi_mpp / st_rela_wsi_mpp / 2)
#                     h_HRST128 = int(h / crop_wsi_mpp / st_rela_wsi_mpp / 2)
#                     gene_save128[h_HRST128, w_HRST128, gene_index] += 1

#                     # 更新SPOT
#                     w_HRST_spot = int(w / crop_wsi_mpp / spotst_rela_wsi_mpp)
#                     h_HRST_spot = int(h / crop_wsi_mpp / spotst_rela_wsi_mpp)
#                     gene_saveSPOT[h_HRST_spot, w_HRST_spot, gene_index] += 1

#             # 保存稀疏矩阵
#             for arr, filename in zip([gene_save256, gene_save128, gene_saveSPOT],
#                                       ['HR_ST_256.npz', 'HR_ST_128.npz', 'spot_ST.npz']):
#                 sparse_matrix = sp_sparse.csr_matrix(arr.reshape(-1, arr.shape[2]))
#                 if "HR" in filename:
#                     sp_sparse.save_npz(os.path.join(block_dir_hrst, filename), sparse_matrix)
#                 else:
#                     sp_sparse.save_npz(os.path.join(block_dir_spot, filename), sparse_matrix)

#         # 释放当前块处理中占用的内存
#         del gene_save256, gene_save128, gene_saveSPOT, transcripts_temp
#         gc.collect()

# a=1
#

# # ##################################
# # ####### validate
# # ##################################
# crop_wsi_mpp=1
# st_rela_wsi_mpp=4
# super_root=r'/home/hanyu/data_preprocess/IHC/20240731Prostate/'
# save_path=r'/home/hanyu/data_preprocess/IHC/20240731Prostate/vis_'+str(crop_wsi_mpp)+'_'+str(st_rela_wsi_mpp)+'/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# else:
#     remove_all_file(save_path)
#
# files=os.listdir(super_root+'/ST_extract_'+str(crop_wsi_mpp)+'_'+str(st_rela_wsi_mpp)+'/')
#
# for i in range(len(files)):
#     name=files[i].split('_')
#     gene_save256=sp_sparse.load_npz(super_root+'/ST_extract_'+str(crop_wsi_mpp)+'_'+str(st_rela_wsi_mpp)+'/'+files[i]).toarray().reshape(int(name[3]),int(name[4]),int(name[5][:-4]))
#     gene_save256=np.transpose(gene_save256,(1,0,2))
#     gene_save256=np.mean(gene_save256,axis=2)
#     gene_save256=cv2.resize(gene_save256,(256,256))
#
#     plt.imsave(save_path+name[0]+'_'+name[1]+'_ST.jpg',gene_save256)
#     WSI=plt.imread(super_root +'WSI_'+str(crop_wsi_mpp)+'/'+name[0]+'_'+name[1]+'.jpg')
#     plt.imsave(save_path+name[0]+'_'+name[1]+'_WSI.jpg',WSI)
#     print(i)
#
# a=2




