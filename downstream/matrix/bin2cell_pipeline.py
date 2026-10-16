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
# 输出 H&E 分割图
#
# in[12] 的画法：把细胞核标签渲染在 he.tiff 上。
# 注：如果报 missing get_cmap()，把 matplotlib 降到 3.9.0 以下。
# ---------------------------------------------------------------------------

rendered = b2c.view_stardist_labels(
    image_path="stardist/he.tiff",
    labels_npz_path="stardist/he.npz",
)
plt.imshow(rendered)
plt.savefig("he_segmentation.png", dpi=300, bbox_inches="tight")
plt.close()

print("done:", os.path.join(OUT_DIR, "he_segmentation.png"))

# 通过修改 prob_thresh 参数可以产出不同的细胞分割结果，
# 细胞分割的HE染色图建议是分辨率为 0.25MPP。
