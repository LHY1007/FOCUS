"""Visium HD 数据分析之 Bin2Cell —— 按公众号教程逐步复现。

代码结构与教程的单元格一一对应（保留 ``# In[n]:`` 编号，便于对照），
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
# In[6]: 校正 bin 的行列偏移
# ---------------------------------------------------------------------------

b2c.destripe(adata)
print(adata)

# 取子图像进行可视化
mask = (
    (adata.obs["array_row"] >= 1450) & (adata.obs["array_row"] <= 1550)
    & (adata.obs["array_col"] >= 250) & (adata.obs["array_col"] <= 450)
)

# ---------------------------------------------------------------------------
# In[7]: 使用 adata.obsm['spatial_cropped'] 进行可视化
# ---------------------------------------------------------------------------

bdata = adata[mask]
sc.pl.spatial(
    bdata,
    color=[None, "n_counts", "n_counts_adjusted"],
    img_key="0.5_mpp",
    basis="spatial_cropped",
)
plt.savefig("destripe_mask.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# In[8]: 使用默认图片进行可视化
# ---------------------------------------------------------------------------

sc.pl.spatial(bdata, color=[None, "n_counts", "n_counts_adjusted"])
plt.savefig("destripe_mask_default.png", dpi=300, bbox_inches="tight")
plt.close()

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
# In[11]: 使用 adata.obsm['spatial_cropped'] 进行可视化 nuclei
# ---------------------------------------------------------------------------

bdata = adata[mask]
bdata = bdata[bdata.obs["labels_he"] > 0]  # 0 means unassigned
bdata.obs["labels_he"] = bdata.obs["labels_he"].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_he"], img_key="0.5_mpp",
              basis="spatial_cropped")
plt.savefig("he_nuclei_mask_test.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# In[12]: 使用HE图片进行可视化 nuclei
# ---------------------------------------------------------------------------

# the label viewer wants a crop of the processed image
# get the corresponding coordinates spanning the subset object
crop = b2c.get_crop(bdata, basis="spatial", spatial_key="spatial_cropped", mpp=mpp)
# if this errors about missing get_cmap(), downgrade matplotlib below 3.9.0
rendered = b2c.view_stardist_labels(
    image_path="stardist/he.tiff",
    labels_npz_path="stardist/he.npz",
    crop=crop,
)
plt.imshow(rendered)
plt.savefig("he_nuclei_mask2.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# In[13]: 扩展 nuclei 为 cell（约1min）
#
# 如果一个 2um bin 与多个细胞核的距离相等，则根据其基因表达谱的相似性
# 将其分配给一个 bin
# max_bin_distance 指定扩展核标签时最大的 bin 数量 (2)
# volume_ratio 如果 max_bin_distance 设置为 None，则根据
#   ceil((volume_ratio**(1/3)-1) * sqrt(n_bins/pi)) 进行扩展 (1)
# k 为每个未分配的 bin 寻找潜在的最近邻的已分配空间坐标 bin 的数量 (5)
# subset_pca 如果设置为 True，则只对于那些与多个细胞核的距离相等的
#   未分配 bins 进行 PCA 表示和决策 (True)
# ---------------------------------------------------------------------------

b2c.expand_labels(
    adata,
    labels_key="labels_he",
    expanded_labels_key="labels_he_expanded",
)
# adata.obs 多了 'labels_he_expanded'

# ---------------------------------------------------------------------------
# In[14]: 使用 adata.obsm['spatial_cropped'] 进行可视化 cell
# ---------------------------------------------------------------------------

bdata = adata[mask]
bdata = bdata[bdata.obs["labels_he_expanded"] > 0]
bdata.obs["labels_he_expanded"] = bdata.obs["labels_he_expanded"].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_he_expanded"], img_key="0.5_mpp",
              basis="spatial_cropped")
plt.savefig("he_cell_mask.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# In[15]: 根据表达矩阵进行细胞分割
#
# 基于 adata.obs['n_counts_adjusted'] 生成高斯模糊图像
# log1p 是否做 log1p-transform（False）
# sigma 使用 skimage.filters.gaussian() 对最终图像进行高斯模糊处理，
#   sigma 值决定模糊的程度 (None)
# ---------------------------------------------------------------------------

b2c.grid_image(adata, "n_counts_adjusted", mpp=mpp, sigma=5,
               save_path="stardist/gex.tiff")

# ---------------------------------------------------------------------------
# In[16]: 使用 StarDist's fluorescence model 进行细胞分割（约10min）
#
# 由于网络原因会下载模型失败，多次尝试即可
# ---------------------------------------------------------------------------

b2c.stardist(
    image_path="stardist/gex.tiff",
    labels_npz_path="stardist/gex.npz",
    stardist_model="2D_versatile_fluo",
    prob_thresh=0.05,
    nms_thresh=0.5,
)

# ---------------------------------------------------------------------------
# In[17]: 转录本分配
# ---------------------------------------------------------------------------

b2c.insert_labels(
    adata,
    labels_npz_path="stardist/gex.npz",
    basis="array",
    mpp=mpp,
    labels_key="labels_gex",
)
print(adata)
# adata.obs 多了 'labels_gex'

# ---------------------------------------------------------------------------
# In[18]: 使用 adata.obsm['spatial_cropped'] 进行可视化 cell
# ---------------------------------------------------------------------------

bdata = adata[mask]
bdata = bdata[bdata.obs["labels_gex"] > 0]
bdata.obs["labels_gex"] = bdata.obs["labels_gex"].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_gex"], img_key="0.5_mpp",
              basis="spatial_cropped")
plt.savefig("gex_cell_mask.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# In[19]: 使用 gex.tiff 进行可视化 cell
# ---------------------------------------------------------------------------

# the label viewer wants a crop of the processed image
# get the corresponding coordinates spanning the subset object
crop = b2c.get_crop(bdata, basis="array", mpp=mpp)
# if this errors about missing get_cmap(), downgrade matplotlib below 3.9.0
rendered = b2c.view_stardist_labels(
    image_path="stardist/gex.tiff",
    labels_npz_path="stardist/gex.npz",
    crop=crop,
)
plt.imshow(rendered)
plt.savefig("gex_cell_mask2.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# In[20]: 用基于表达量进行细胞分割的结果填补 HE 细胞核分割结果中未识别的细胞
# ---------------------------------------------------------------------------

b2c.salvage_secondary_labels(
    adata,
    primary_label="labels_he_expanded",
    secondary_label="labels_gex",
    labels_key="labels_joint",
)
print(adata)
# adata.obs 多了 'labels_joint', 'labels_joint_source'

# ---------------------------------------------------------------------------
# In[21]: 使用 adata.obsm['spatial_cropped'] 进行可视化 cell
# ---------------------------------------------------------------------------

bdata = adata[mask]
bdata = bdata[bdata.obs["labels_joint"] > 0]
bdata.obs["labels_joint"] = bdata.obs["labels_joint"].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_joint_source", "labels_joint"],
              img_key="0.5_mpp", basis="spatial_cropped")
plt.savefig("he_gex_mask_bin.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# In[22]: 将同一个 cell 中的 bin 进行合并
# ---------------------------------------------------------------------------

cdata = b2c.bin_to_cell(adata, labels_key="labels_joint",
                        spatial_keys=["spatial", "spatial_cropped"])
print(cdata)

# ---------------------------------------------------------------------------
# In[23]: 使用 adata.obsm['spatial_cropped'] 进行可视化 cell
# ---------------------------------------------------------------------------

cell_mask = (
    (cdata.obs["array_row"] >= 1450) & (cdata.obs["array_row"] <= 1550)
    & (cdata.obs["array_col"] >= 250) & (cdata.obs["array_col"] <= 450)
)
ddata = cdata[cell_mask]
sc.pl.spatial(ddata, color=["bin_count", "labels_joint_source"],
              img_key="0.5_mpp", basis="spatial_cropped")
plt.savefig("he_gex_mask_cell.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# In[24]: 保存数据
# ---------------------------------------------------------------------------

cdata.write_h5ad("m_brain_b2c.h5ad")
adata.write_h5ad("m_brain_2um.h5ad")

# 通过修改 prob_thresh 参数可以产出不同的细胞分割结果，
# 细胞分割的HE染色图建议是分辨率为 0.25MPP。
