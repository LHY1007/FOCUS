"""Visium HD 数据分析之 Bin2Cell —— H&E 细胞核分割。

只保留到产出 H&E 分割图为止的部分，其余（destripe、转录本分配、基于表达量
的二次分割、salvage、bin_to_cell）都不需要。

代码结构与教程的单元格对应（保留 ``# In[n]:`` 编号，便于对照），
只把路径换成本项目的数据位置。

    python bin2cell_pipeline.py
"""

import os

import matplotlib.pyplot as plt
import scanpy as sc

import bin2cell as b2c

# ---------------------------------------------------------------------------
# In[2]: 地址设置
# ---------------------------------------------------------------------------

# 本项目的数据位置
SAMPLE_DIR = "/data_temp/mnt/raid5/qnf/bone_project/project_bone/ST/AXB-1-2-A1/16-4-NC_SR"

OUT_DIR = "/data_temp/mnt/raid5/qnf/bone_project/project_bone/bin2cell_run/16-4-NC"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
os.makedirs("stardist", exist_ok=True)

path = SAMPLE_DIR + "/binned_outputs/square_002um/"
source_image_path = "/data_temp/mnt/raid5/qnf/bone_project/project_bone/HE_image/1-2-TIFF.tif"
spaceranger_image_path = SAMPLE_DIR + "/binned_outputs/square_002um/spatial"

# ---------------------------------------------------------------------------
# In[3]: 导入HD表达数据（2um）和高清HE图片
# ---------------------------------------------------------------------------

adata = b2c.read_visium(
    path,
    source_image_path=source_image_path,
    spaceranger_image_path=spaceranger_image_path,
)
adata.var_names_make_unique()
print(adata)

# ---------------------------------------------------------------------------
# In[4]: 过滤低质量bins
# ---------------------------------------------------------------------------

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_counts=1)
print(adata)

# ---------------------------------------------------------------------------
# In[5]: 基于高清图像保存低分辨率图像并存入 adata.obsm['spatial_cropped']
# ---------------------------------------------------------------------------

mpp = 0.5  # 每像素微米数
b2c.scaled_he_image(adata, mpp=mpp, save_path="stardist/he.tiff")
print(adata)

# ---------------------------------------------------------------------------
# In[9]: 细胞核分割（约3min）
#
# block_size   StarDist 在 predict_instances_big() 中使用的参数，
#              指定作为单个瓦片处理的图像的正方形边长（4096）
# min_overlap  StarDist 在 predict_instances_big() 中使用的参数，
#              指定相邻瓦片之间的最小重叠区域（128）
# context      StarDist 在 predict_instances_big() 中使用的参数，
#              指定在每个瓦片边界上要丢弃的图像上下文量
# prob_thresh  控制调用对象的严格性，值越小分割的细胞越多
# nms_thresh   非最大抑制中定义了预测对象被视为重叠并合并成单一标签的阈值
# ---------------------------------------------------------------------------

b2c.stardist(
    image_path="stardist/he.tiff",
    labels_npz_path="stardist/he.npz",
    stardist_model="2D_versatile_he",
    prob_thresh=0.01,
)

# ---------------------------------------------------------------------------
# In[10]: 转录本分配
#
# 把细胞核标签映射到 bin 上，得到 adata.obs['labels_he']。
# expand_labels 需要先有这一列，所以这步不能省。
# ---------------------------------------------------------------------------

b2c.insert_labels(
    adata,
    labels_npz_path="stardist/he.npz",
    basis="spatial",
    spatial_key="spatial_cropped",
    mpp=mpp,
    labels_key="labels_he",  # .obs key to store the labels under.
)
print(adata)
# adata.uns 多了 'bin2cell'
# adata.obs 多了 'labels_he'

# ---------------------------------------------------------------------------
# In[13]: mask 后处理 —— 把细胞核扩张成细胞（约1min）
#
# 这一步就是补充材料里的 "masks were isotropically dilated in a
# non-overlapping manner to approximate full-cell boundaries"：bin2cell 按
# 距离把未分配的 bin 归给最近的核，距离相同时再按表达谱相似性决定，
# 因此不会有两个细胞抢同一个 bin。
#
# max_bin_distance 扩展核标签时最大的 bin 数量 (2)
# volume_ratio 若 max_bin_distance 为 None，则按
#   ceil((volume_ratio**(1/3)-1) * sqrt(n_bins/pi)) 扩展 (1)
# k 为每个未分配的 bin 寻找最近邻的已分配 bin 的数量 (5)
# subset_pca 只对与多个核等距的未分配 bins 做 PCA 决策 (True)
# ---------------------------------------------------------------------------

b2c.expand_labels(
    adata,
    labels_key="labels_he",
    expanded_labels_key="labels_he_expanded",
)
print(adata)
# adata.obs 多了 'labels_he_expanded'

# ---------------------------------------------------------------------------
# 输出 H&E 分割图
#
# in[12] / in[14] 的画法：把标签渲染在 he.tiff 上。
# 注：如果报 missing get_cmap()，把 matplotlib 降到 3.9.0 以下。
# ---------------------------------------------------------------------------

rendered = b2c.view_stardist_labels(
    image_path="stardist/he.tiff",
    labels_npz_path="stardist/he.npz",
)
plt.imshow(rendered)
plt.savefig("he_segmentation.png", dpi=300, bbox_inches="tight")
plt.close()

# 扩张后的细胞标签叠加到 HE 上
bdata = adata[adata.obs["labels_he_expanded"] > 0].copy()
bdata.obs["labels_he_expanded"] = bdata.obs["labels_he_expanded"].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_he_expanded"], img_key="0.5_mpp",
              basis="spatial_cropped", show=False)
plt.savefig("he_cell_mask.png", dpi=300, bbox_inches="tight")
plt.close()

print("done:", OUT_DIR)

# 通过修改 prob_thresh 参数可以产出不同的细胞分割结果，
# 细胞分割的HE染色图建议是分辨率为 0.25MPP。
