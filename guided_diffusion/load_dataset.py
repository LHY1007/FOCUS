"""Multi-class patch dataset for FOCUS.

`Data_all` reads one directory per tissue type, builds the low-resolution ST
input, the paired H&E patches, the scGPT-spatial embeddings, the scRNA-seq
reference used by the OT fusion and the spatial co-expression matrix that
backs the gene memory bank.  `load_data` is the factory used by `train.py`,
`train_part.py` and `sample.py`.
"""

import os
import hashlib
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch

import scipy.sparse as sp

from transformers import AutoTokenizer, AutoModel

import scanpy as sc

BIOBERT_V11_SHA256 = '8e87cb27d889186c442feff2bea53f8cce2c5fc24d4bc88bf08ccd3821b138dc'



# =========================
# 工具函数（保持）
# =========================
def np_norm(inp):
    max_in = np.max(inp)
    min_in = np.min(inp)
    return (inp - min_in) / (max_in - min_in + 1e-8)


def gray_value_of_gene(gene_class, gene_order):
    gene_order = list(gene_order)
    Index = gene_order.index(gene_class)
    interval = 255 / len(gene_order)
    value = Index * interval
    return int(value)


# =========================
# 顶层入口（保持接口不变）
# =========================
def load_data(data_root, dataset_use, status, SR_times, gene_num, all_gene,
              gene_order=None, gene_name_order=None, pre_model_dir=None, **options):
    dataset = Data_all(
        data_root, dataset_use, SR_times, status, gene_num, all_gene,
        gene_order, gene_name_order, pre_model_dir=pre_model_dir, **options
    )
    return dataset


class Data_all(Dataset):
    """
    多类（multi-class）多子数据集联合加载：
    - 每个类根目录包含 coexpression_matrix.npy（← 修正路径）
    - 每个类下 sc/ 仅有唯一 .h5ad 作为该类共享 sc 参考
    - 每个 sample 的当前五基因按 gene_order.npy / gene_name.txt 的 offset 选择
    - scRNA 和 co-expression 行均以当前基因名重新排序
    """

    def __init__(self, data_root, dataset_use, SR_times, status, gene_num,
                 all_gene, gene_order, gene_name_order, pre_model_dir,
                 gene_start=0, split_manifest='', split='', reference_cells=256,
                 seed=0, strict_features=True):
        self.data_root = data_root
        self.status = status
        self.gene_num = int(gene_num) if gene_num is not None else 1
        self.gene_start = int(gene_start)
        self.reference_cells, self.seed = int(reference_cells), int(seed)
        self.strict_features = strict_features
        if dataset_use == 'All':
            self.selected_classes = sorted(c for c in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, c)) and any(
                    os.path.isdir(os.path.join(data_root, c, s, 'HR_ST', 'extract'))
                    for s in os.listdir(os.path.join(data_root, c))))
        elif os.path.isdir(os.path.join(data_root, dataset_use)):
            self.selected_classes = [dataset_use]
        else:
            raise ValueError(f"Unknown tissue class: {dataset_use}")

        # ---- 扫描所有类下的 patch 信息，记录所属类
        self.selected_patches = []
        for cls_name in self.selected_classes:
            print(f"Scanning class: {cls_name}")
            cls_patches = self._load_patch_list(self.data_root, cls_name)
            # (cls_name, sub, layer_name, patch_id)
            self.selected_patches.extend([(cls_name, *p) for p in cls_patches])

        if split_manifest:
            from .splits import read_manifest
            assignments = read_manifest(split_manifest)
            if not split:
                raise ValueError('Specify split when using split_manifest')
            for cls_name, sub, _, _ in self.selected_patches:
                if (cls_name, sub) not in assignments:
                    raise ValueError(f'Sample absent from patient manifest: {cls_name}/{sub}')
            self.selected_patches = [p for p in self.selected_patches
                                     if assignments[p[:2]]['split'] == split]
            self.selected_classes = sorted(set(p[0] for p in self.selected_patches))
        elif strict_features:
            raise ValueError('Full training/inference requires split_manifest and split')

        if status == 'Test':
            self._sort_patches()

        # ---- 每类加载 sc.h5ad（仅一个）
        self.sc_reference = {}
        for cls_name in self.selected_classes:
            self.sc_reference[cls_name] = self._load_sc_reference(cls_name)

        # ---- 每子数据集的当前 gene_order.npy / gene_name.txt 分组
        self.sub_gene_info = self._load_subdataset_gene_info()

        # ---- 加载每类共表达矩阵（修正路径：<root>/<cls>/coexpression_matrix.npy）
        self._load_coexpression(self.data_root)

        # ---- 加载 ST / spot / WSI / mask / sc 索引
        self._load_data(self.data_root)

        # ---- 归一化
        self._normalize_data()

        # ---- BERT 元数据
        self._load_bert_model(pre_model_dir)
        self._embed_metadata()

        # ---- gene index maps（保持）
        self._precompute_gene_index_maps()

        print(f"[{status}] Loaded {len(self.selected_patches)} patches from {len(self.selected_classes)} classes.")

    # ===================================================== #
    # Patch 扫描
    # 目录结构（示例）：
    # <root>/<cls>/<sub>/HR_ST/extract/<layer>/<patch_id>/HR_ST_256.npz
    # ===================================================== #
    def _load_patch_list(self, data_root, cls_name):
        patches = []
        root = os.path.join(data_root, cls_name)
        if not os.path.isdir(root):
            print(f"Warning: missing class dir: {root}")
            return patches

        for sub in sorted(os.listdir(root)):
            sub_path = os.path.join(root, sub)
            if not os.path.isdir(sub_path) or sub == 'sc':
                continue
            extract_base = os.path.join(sub_path, 'HR_ST', 'extract')
            if not os.path.isdir(extract_base):
                continue
            for layer_name in sorted(os.listdir(extract_base)):
                layer_dir = os.path.join(extract_base, layer_name)
                if not os.path.isdir(layer_dir):
                    continue
                for patch_id in sorted(os.listdir(layer_dir)):
                    patches.append((sub, layer_name, patch_id))
        return patches

    def _sort_patches(self):
        # selected_patches: (cls_name, sub, layer, patch_id)
        # 这里按 (cls, sub, layer, patch_id) 字典序排序，或可按 patch_id 自定义规则
        self.selected_patches.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    # ===================================================== #
    # 读取每类 sc 文件（仅一个 .h5ad；归一化前不截断基因）
    # ===================================================== #
    def _load_sc_reference(self, cls_name):
        sc_dir = os.path.join(self.data_root, cls_name, 'sc')
        from pathlib import Path
        files = sorted(Path(sc_dir).glob('*.h5ad'))
        if len(files) != 1:
            raise ValueError(f'{sc_dir}: expected exactly one tissue-matched reference h5ad')
        adata = sc.read_h5ad(files[0])
        adata.var_names = adata.var_names.astype(str)
        if not adata.var_names.is_unique:
            raise ValueError(f'{files[0]}: duplicate gene names; resolve aliases before loading')
        counts = np.asarray(adata.X.sum(axis=1)).ravel()
        adata = adata[counts > 0].copy()
        if not adata.n_obs:
            raise ValueError(f'{files[0]}: no nonzero cells')
        if self.reference_cells <= 0 or adata.n_obs < self.reference_cells:
            raise ValueError(f'{files[0]} has {adata.n_obs} cells; set reference_cells between 1 and that count')
        rng = np.random.default_rng(self.seed)
        indices = np.sort(rng.choice(adata.n_obs, self.reference_cells, replace=False))
        adata = adata[indices].copy()
        # Normalize over the complete reference gene universe, then select genes.
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f'{cls_name}: reference={files[0]}, cells={adata.n_obs}, genes={adata.n_vars}, seed={self.seed}')
        return adata

    # ===================================================== #
    # 读取子数据集当前分组的基因编号与名称
    # ===================================================== #
    def _load_subdataset_gene_info(self):
        info = {}
        for cls_name, sub, _, _ in self.selected_patches:
            key = (cls_name, sub)
            if key in info:
                continue
            base = os.path.join(self.data_root, cls_name, sub)
            order = np.load(os.path.join(base, 'gene_order.npy')).astype(int)
            names = np.loadtxt(os.path.join(base, 'gene_name.txt'), dtype=str, ndmin=1)
            if len(names) != len(order) or len(set(names.tolist())) != len(names):
                raise ValueError(f'{base}: gene_name.txt must uniquely name each ordered gene')
            end = self.gene_start + self.gene_num
            if self.gene_start < 0 or end > len(order):
                raise ValueError(f'{base}: requested genes [{self.gene_start}:{end}], available {len(order)}')
            selected_names = names[self.gene_start:end].tolist()
            reference = self.sc_reference[cls_name]
            missing = [n for n in selected_names if n not in reference.var_names]
            if missing:
                raise ValueError(f'{base}: selected ST genes absent from reference: {missing}')
            aligned = reference[:, selected_names].X
            if sp.issparse(aligned):
                aligned = aligned.toarray()
            info[key] = {'idxN': order[self.gene_start:end], 'namesN': selected_names,
                         'reference': np.asarray(aligned, dtype=np.float32)}
        return info

    # ===================================================== #
    # 加载每类共表达矩阵（修正路径）
    # <root>/<cls>/coexpression_matrix.npy
    # ===================================================== #
    def _load_coexpression(self, root):
        """Keep the supplied matrix intact and map its rows by explicit gene names."""
        self.co_expression_dict = {}
        self.co_expression_names = {}
        for cls_name in self.selected_classes:
            coex_path = os.path.join(root, cls_name, 'coexpression_matrix.npy')
            names_path = os.path.join(root, cls_name, 'coexpression_matrix.genes.txt')
            if not os.path.isfile(coex_path) or not os.path.isfile(names_path):
                raise FileNotFoundError(f'{cls_name}: require coexpression_matrix.npy and its ordered .genes.txt sidecar')
            matrix = np.load(coex_path, mmap_mode='r')
            names = np.loadtxt(names_path, dtype=str, ndmin=1).tolist()
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(names) != matrix.shape[0] or len(set(names)) != len(names):
                raise ValueError(f'{cls_name}: co-expression matrix and unique gene names must have matching square dimensions')
            self.co_expression_dict[cls_name] = matrix
            self.co_expression_names[cls_name] = {name: i for i, name in enumerate(names)}

    def _coexpression_features(self, cls_name, gene_names):
        """Current gene rows, pooled to a fixed-width memory vector per gene."""
        import torch.nn.functional as F
        names = self.co_expression_names[cls_name]
        missing = [name for name in gene_names if name not in names]
        if missing:
            raise ValueError(f'{cls_name}: selected ST genes absent from co-expression matrix: {missing}')
        rows = np.asarray(self.co_expression_dict[cls_name][[names[name] for name in gene_names]], dtype=np.float32)
        if not np.isfinite(rows).all():
            raise ValueError(f'{cls_name}: nonfinite selected co-expression rows')
        return F.adaptive_avg_pool1d(torch.from_numpy(rows)[None], 64).squeeze(0)

    # ===================================================== #
    # 加载 ST / spot / WSI / mask / sc
    # ===================================================== #
    def _load_data(self, root):
        sr_list, spot_list, wsi5120_list, wsi320_list, wsimask_list, sc_list = [], [], [], [], [], []
        scgpt_list, prehe_list, complexity_list = [], [], []
        loaded_patches = []

        for cls_name, sub, layer_name, patch_id in self.selected_patches:
            base = os.path.join(root, cls_name, sub)
            key = (cls_name, sub)
            if key not in self.sub_gene_info:
                # 若该子数据集未提供基因文件则跳过该 patch
                print(f"Warning: skipping patch without gene info: {(cls_name, sub, layer_name, patch_id)}")
                continue

            idxN = self.sub_gene_info[key]['idxN']

            paths = {
                'hr': os.path.join(base, 'HR_ST', 'extract', layer_name, patch_id, 'HR_ST_256.npz'),
                'spot': os.path.join(base, 'spot_ST', 'extract', layer_name, patch_id, 'spot_ST.npz'),
                'wsi5120': os.path.join(base, 'WSI', 'extract', layer_name, patch_id, '5120_to256.npy'),
                'wsi320': os.path.join(base, 'WSI', 'extract', layer_name, patch_id, '320_to16.npy'),
                'mask': os.path.join(base, 'WSI', 'extract', layer_name, patch_id, 'cell_mask.npy'),
            }

            if not (os.path.exists(paths['hr']) and os.path.exists(paths['spot']) and
                    os.path.exists(paths['wsi5120']) and os.path.exists(paths['wsi320']) and
                    os.path.exists(paths['mask'])):
                if getattr(self, 'strict_features', False):
                    raise FileNotFoundError(f'Missing required patch inputs: {paths}')
                print(f"Warning: missing files for patch: {paths}")
                continue

            # 稀疏矩阵读取与截取
            sr = sp.load_npz(paths['hr'])[:, idxN].toarray().reshape(256, 256, -1).transpose(2, 0, 1)
            spot = sp.load_npz(paths['spot'])[:, idxN].toarray().reshape(26, 26, -1).transpose(2, 0, 1)

            # 影像加载
            wsi5120 = np.load(paths['wsi5120']).transpose(2, 0, 1)
            # Rebuild the row-major cellular grid; historical files were entropy-sorted.
            wsi320 = wsi5120.reshape(3, 16, 16, 16, 16).transpose(1, 3, 0, 2, 4).reshape(256, 3, 16, 16)
            wsimask = np.load(paths['mask']).transpose(2, 0, 1)
            if not len(wsimask):
                wsimask = np.zeros((1, 256, 256), dtype=np.float32)

            sr_list.append(sr.astype(np.float32))
            spot_list.append(spot.astype(np.float32))
            wsi5120_list.append(wsi5120.astype(np.float32))
            wsi320_list.append(wsi320.astype(np.float32))
            wsimask_list.append(wsimask.astype(np.float32))

            # 记录所属类的 sc 参考（直接存 adata 引用）
            sc_list.append(self.sub_gene_info[key]["reference"])

            # ---------- 读取 scGPT ----------
            scgpt_dir = os.path.join(base, 'spot_ST', 'extract', layer_name, patch_id, 'scgpt_data')
            gene_names = self.sub_gene_info[key]['namesN']
            scgpt_embed = load_scgpt_embedding(scgpt_dir, gene_names, gene_start=self.gene_start)
            scgpt_list.append(scgpt_embed)

            # ---------- 新增：读取 pre_he ----------
            pre_he_path = os.path.join(base, 'WSI', 'extract', layer_name, patch_id, 'pre_he.npy')
            pre_he = load_pre_he(pre_he_path)
            if pre_he is None:
                raise FileNotFoundError(f'Missing authentic histology features: {pre_he_path}')
            prehe_list.append(pre_he)
            complexity_path = os.path.join(base, 'WSI', 'extract', layer_name, patch_id, 'complexity_grid.npy')
            if not os.path.isfile(complexity_path):
                raise FileNotFoundError(f'{complexity_path}: score registered native H&E and cell labels')
            complexity = np.load(complexity_path).astype(np.float32)
            if complexity.shape != (16, 16) or not np.isfinite(complexity).all() or np.any((complexity < 0) | (complexity > 1)):
                raise ValueError(f'{complexity_path}: expected finite [16,16] scores in [0,1]')
            complexity_list.append(complexity)
            loaded_patches.append((cls_name, sub, layer_name, patch_id))
        # 转为数组
        self.selected_patches = loaded_patches
        self.SR_ST_all = np.asarray(sr_list, dtype=np.float32)
        self.spot_ST_all = np.asarray(spot_list, dtype=np.float32)
        self.WSI_5120_all = np.asarray(wsi5120_list, dtype=np.float32)
        self.WSI_320_all = np.asarray(wsi320_list, dtype=np.float32)
        self.WSI_mask_all = np.asarray(wsimask_list, dtype=np.float32)
        self.sc_all = sc_list
        self.scgpt_list = scgpt_list
        self.prehe_list = prehe_list
        self.complexity_list = complexity_list

        if len(self.SR_ST_all) == 0:
            raise ValueError("No valid patches loaded. Check data_root, tissue class and required files.")

    # ===================================================== #
    # 归一化（保持）
    # ===================================================== #
    def _normalize_data(self):
        for i in range(self.spot_ST_all.shape[0]):
            data = self.spot_ST_all[i]
            mins = data.min(axis=(1, 2), keepdims=True)
            maxs = data.max(axis=(1, 2), keepdims=True)
            denom = maxs - mins + 1e-8
            self.spot_ST_all[i] = (data - mins) / denom
        patch_max = np.max(self.SR_ST_all, axis=(1, 2, 3))
        patch_max[patch_max == 0] = 1.0
        self.patch_scale = patch_max.astype(np.float32)
        self.SR_ST_all = self.SR_ST_all / patch_max[:, None, None, None]

    # ===================================================== #
    # BERT 元数据（保持）
    # ===================================================== #
    def _load_bert_model(self, pre_model_dir):
        bert_dir = os.path.join(pre_model_dir, 'bert')
        weight_path = os.path.join(bert_dir, 'pytorch_model.bin')
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f'{weight_path}: supply the pinned dmis-lab/biobert-v1.1 weights')
        with open(weight_path, 'rb') as weights:
            actual_sha = hashlib.file_digest(weights, 'sha256').hexdigest()
        if actual_sha != BIOBERT_V11_SHA256:
            raise ValueError(f'{weight_path}: does not match the pinned BioBERT v1.1 artifact')
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(bert_dir, local_files_only=True)
        if (self.model.config.model_type != 'bert' or self.model.config.hidden_size != 768 or
                self.model.config.vocab_size != 28996 or len(self.tokenizer) != 28996):
            raise ValueError(f'{bert_dir}: expected the pinned BioBERT v1.1 model and tokenizer')
        self.model.eval()
        self.model.requires_grad_(False)

    def _embed_metadata(self):
        from pathlib import Path
        self.metadata_features = {}
        for key, gene_info in self.sub_gene_info.items():
            cls_name, sub = key
            path = Path(self.data_root) / cls_name / sub / 'metadata_prompt.txt'
            if not path.is_file():
                raise FileNotFoundError(f'{cls_name}/{sub}: provide metadata_prompt.txt with actual platform, species and tissue')
            prompt = path.read_text(encoding='utf-8').strip()
            if not prompt:
                raise ValueError(f'Empty metadata prompt: {path}')
            # Put the current group first so tokenizer truncation cannot drop it.
            prompt = 'Target genes: ' + ', '.join(gene_info['namesN']) + '. ' + prompt
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                outputs = self.model(**inputs)
                self.metadata_features[key] = outputs.pooler_output.squeeze(0)
        # The frozen text encoder is needed only during preparation.
        del self.model
        del self.tokenizer

    # ===================================================== #
    # Gene index maps（保持）
    # ===================================================== #
    def _precompute_gene_index_maps(self):
        N = len(self.SR_ST_all)
        self.gene_index_maps_all = np.zeros((N, self.gene_num, 256, 256), dtype=np.float32)
        for i in range(N):
            for j in range(self.gene_num):
                self.gene_index_maps_all[i, j] = np.full((256, 256), j / self.gene_num, dtype=np.float32)

    # ===================================================== #
    # Dataset 接口
    # ===================================================== #
    def __len__(self):
        return len(self.SR_ST_all)

    def __getitem__(self, idx):
        """
        返回：
        - SR_ST[ C,256,256 ], spot_ST[ C,26,26 ], WSI_5120[ 3,256,256 ], WSI_320[ T,3,16,16 ]
        - gene_index_maps[ C,256,256 ], metadata_feature[768], patch_scale(float tensor)
        - coexpression memory[ C,64 ]（当前基因名对应的矩阵行）
        - WSI_mask[ M,256,256 ]
        - sc_item: 当前基因顺序的参考细胞表达张量
        - complexity: 原生分辨率形态复杂度网格 [16,16]
        """
        cls_name = self.selected_patches[idx][0]
        sample_key = self.selected_patches[idx][:2]
        co_expression = self._coexpression_features(cls_name, self.sub_gene_info[sample_key]['namesN'])

        patch_scale = torch.tensor(self.patch_scale[idx], dtype=torch.float32)
        sc_item = torch.as_tensor(self.sc_all[idx], dtype=torch.float32)

        return (
            self.SR_ST_all[idx],
            self.spot_ST_all[idx],
            self.WSI_5120_all[idx],
            self.WSI_320_all[idx],
            self.sub_gene_info[sample_key]['idxN'].astype(np.float32),
            self.gene_index_maps_all[idx],
            self.metadata_features[self.selected_patches[idx][:2]],
            patch_scale,
            co_expression,
            self.WSI_mask_all[idx],
            sc_item,
            self.scgpt_list[idx],
            self.prehe_list[idx],
            self.complexity_list[idx]
        )

def load_scgpt_embedding(scgpt_dir, gene_names, gene_start=0):
    """One authentic spot embedding map; validate its selected gene names."""
    from pathlib import Path
    base = Path(scgpt_dir)
    first, last = gene_start + 1, gene_start + len(gene_names)
    path = base / f'{first}to{last}.npy'
    if not path.is_file():
        raise FileNotFoundError(f'{path}: export actual scGPT-spatial features for this gene group')
    names_path = path.with_suffix('.genes.txt')
    if not names_path.is_file():
        raise FileNotFoundError(f'{names_path}: provide gene names in selected ST order')
    names = np.loadtxt(names_path, dtype=str, ndmin=1).tolist()
    if names != list(gene_names):
        raise ValueError(f'{names_path}: feature gene order differs from ST target order')
    feature = np.load(path).astype(np.float32)
    if feature.shape != (512, 26, 26) or not np.isfinite(feature).all():
        raise ValueError(f'{path}: expected finite [512,26,26] features')
    if not np.any(feature):
        raise ValueError(f'{path}: zero placeholder features are not accepted for full training')
    return feature


def load_pre_he(prehe_path):
    if not os.path.isfile(prehe_path):
        return None
    value = np.load(prehe_path).astype(np.float32)
    if value.shape != (257, 1536) or not np.isfinite(value).all() or not np.any(value):
        raise ValueError(f'{prehe_path}: expected authentic [257,1536] tissue and cellular tile vectors')
    return value


def collate_patches(batch):
    """Pad only the cell-mask axis; references have configured equal cell count."""
    from torch.utils.data._utils.collate import default_collate
    count = max(item[9].shape[0] for item in batch)
    items = []
    for sample in batch:
        sample = list(sample)
        sample[9] = np.pad(sample[9], ((0, count - sample[9].shape[0]), (0, 0), (0, 0)))
        items.append(tuple(sample))
    return default_collate(items)
