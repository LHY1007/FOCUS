import pathlib
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
Image.MAX_IMAGE_PIXELS = 2500000000000
import numpy as np
import json
import pandas as pd
import cv2
import os
import collections
import scipy.sparse as sp_sparse
#from pyometiff import OMETIFFReader
import torch.nn.functional as F
import torch
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
from numpy.linalg import inv
import os
import scipy
import anndata
import sklearn
import torch
import random
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)
def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag

# CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix'])
# def get_matrix_from_h5(filename):
#     import tables
#     with tables.open_file(filename, 'r') as f:
#         mat_group = f.get_node(f.root, 'matrix')
#         barcodes = f.get_node(mat_group, 'barcodes').read()
#         data = getattr(mat_group, 'data').read()
#         indices = getattr(mat_group, 'indices').read()
#         indptr = getattr(mat_group, 'indptr').read()
#         shape = getattr(mat_group, 'shape').read()
#         matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
#
#         feature_ref = {}
#         feature_group = f.get_node(mat_group, 'features')
#         feature_ids = getattr(feature_group, 'id').read()
#         feature_names = getattr(feature_group, 'name').read()
#         feature_types = getattr(feature_group, 'feature_type').read()
#         feature_ref['id'] = feature_ids
#         feature_ref['name'] = feature_names
#         feature_ref['feature_type'] = feature_types
#         tag_keys = getattr(feature_group, '_all_tag_keys').read()
#         for key in tag_keys:
#             feature_ref[key] = getattr(feature_group, 'genome').read()
#         return CountMatrix(feature_ref, barcodes, matrix)
import h5py
import scipy.sparse as sp_sparse

class CountMatrix:
    def __init__(self, feature_ref, barcodes, matrix):
        self.feature_ref = feature_ref
        self.barcodes = barcodes
        self.matrix = matrix

    def __getitem__(self, key):
        # Delegate subscripting to the matrix attribute
        return self.matrix[key]

    def __repr__(self):
        return f"CountMatrix(matrix_shape={self.matrix.shape}, features={len(self.feature_ref)}, barcodes={len(self.barcodes)})"



def get_matrix_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        # Access the 'matrix' group
        mat_group = f['matrix']

        # Read barcodes
        barcodes = mat_group['barcodes'][()]

        # Read sparse matrix components
        data = mat_group['data'][()]
        indices = mat_group['indices'][()]
        indptr = mat_group['indptr'][()]
        shape = tuple(mat_group['shape'][()])  # Ensure shape is a tuple

        # Create the sparse CSC matrix
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

        # Access the 'features' group
        feature_ref = {}
        feature_group = mat_group['features']

        feature_ref['id'] = feature_group['id'][()]
        feature_ref['name'] = feature_group['name'][()]
        feature_ref['feature_type'] = feature_group['feature_type'][()]

        # Read additional tag keys
        tag_keys = feature_group['_all_tag_keys'][()]
        for key in tag_keys:
            feature_ref[key] = feature_group[key][()]

        # Return the result as a CountMatrix object
        return CountMatrix(feature_ref, barcodes, matrix)

def _invertAffineTransform(matrix):

    matrix = np.concatenate([matrix, np.array([0, 0, 1], dtype=matrix.dtype)[None]])
    return inv(matrix)[:2]


def __warpAffine(x, matrix, dsize=None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    """
    Custom implementation of cv2.warpAffine function.

    Parameters:
    x : ndarray
        Input image.
    matrix : ndarray
        2x3 affine transformation matrix.
    dsize : tuple, optional
        Size of the output image (width, height). Default is input image size.
    flags : int, optional
        Interpolation method. Default is cv2.INTER_LINEAR.
    borderMode : int, optional
        Pixel extrapolation method. Default is cv2.BORDER_CONSTANT.
    borderValue : int or tuple, optional
        Value used in case of a constant border. Default is 0.

    Returns:
    output : ndarray
        The transformed image.
    """
    # Set output size to input image size if not provided
    dsize = dsize or (x.shape[1], x.shape[0])

    # If WARP_INVERSE_MAP flag is not set, invert the matrix
    if flags & cv2.WARP_INVERSE_MAP == 0:
        matrix = _invertAffineTransform(matrix)

    # Create a grid of coordinates in the destination image
    grid_x, grid_y = np.meshgrid(np.arange(dsize[0]), np.arange(dsize[1]),copy=False)

    # Calculate the corresponding source coordinates using the affine transform matrix
    src_x = (matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2])
    src_y = (matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2])

    del grid_x
    del grid_y
    # Interpolation (bilinear interpolation is implemented here)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(np.int32)
    y1 = y0 + 1

    # Clip the coordinates to be within valid bounds
    x0_clip = np.clip(x0, 0, x.shape[1] - 1)
    x1_clip = np.clip(x1, 0, x.shape[1] - 1)
    y0_clip = np.clip(y0, 0, x.shape[0] - 1)
    y1_clip = np.clip(y1, 0, x.shape[0] - 1)

    # Get the pixel values at the 4 surrounding points
    Ia = x[y0_clip, x0_clip]
    Ib = x[y1_clip, x0_clip]
    Ic = x[y0_clip, x1_clip]
    Id = x[y1_clip, x1_clip]

    # Calculate the weights for interpolation
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (x1 - src_x) * (src_y - y0)
    wc = (src_x - x0) * (y1 - src_y)
    wd = (src_x - x0) * (src_y - y0)

    # Expand dimensions of weights to match the input image shape
    wa = wa[:, :, None]
    wb = wb[:, :, None]
    wc = wc[:, :, None]
    wd = wd[:, :, None]

    # Compute the output image using the weights
    output = wa * Ia + wb * Ib + wc * Ic + wd * Id

    # Handle borders
    if borderMode == cv2.BORDER_REPLICATE:
        mask = (src_x < 0) | (src_x >= x.shape[1]) | (src_y < 0) | (src_y >= x.shape[0])
        output[mask] = borderValue

    return output.astype(x.dtype)

def __warpAffine2(img_array, matrix, dsize=None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    matrix=matrix.astype(np.float32)


    # Set output size to input image size if not provided
    dsize = dsize or (img_array.shape[1], img_array.shape[0])

    # If WARP_INVERSE_MAP flag is not set, invert the matrix
    if flags & cv2.WARP_INVERSE_MAP == 0:
        matrix = _invertAffineTransform(matrix)

    # Create a grid of coordinates in the destination image
    grid_x, grid_y = np.meshgrid(np.arange(dsize[0]), np.arange(dsize[1]), copy=False)

    src_x =(matrix[0, 0] * grid_x).astype(np.int32)+(matrix[0, 1] * grid_y).astype(np.int32)+matrix[0, 2].astype(np.int32)
    src_y = (matrix[1, 0] * grid_x).astype(np.int32)+(matrix[1, 1] * grid_y).astype(np.int32)+matrix[1, 2].astype(np.int32)

    del grid_x
    del grid_y

    # Interpolation (bilinear interpolation is implemented here)
    # x0 = np.floor(src_x).astype(np.int32)
    # y0 = np.floor(src_y).astype(np.int32)

    # Clip the coordinates to be within valid bounds
    # x0_clip = np.clip(x0, 0, img_array.shape[1] - 1)
    # x1_clip = np.clip(x0 + 1, 0, img_array.shape[1] - 1)
    # y0_clip = np.clip(y0, 0, img_array.shape[0] - 1)
    # y1_clip = np.clip(y0 + 1, 0, img_array.shape[0] - 1)
    # x0_clip = np.clip(src_x, 0, img_array.shape[1] - 1)
    # x1_clip = np.clip(src_x + 1, 0, img_array.shape[1] - 1)
    # y0_clip = np.clip(src_y, 0, img_array.shape[0] - 1)
    # y1_clip = np.clip(src_y + 1, 0, img_array.shape[0] - 1)
    # Get the pixel values at the 4 surrounding points
    # Ia = img_array[y0_clip, x0_clip]
    # Ib = img_array[y1_clip, x0_clip]
    # Ic = img_array[y0_clip, x1_clip]
    # Id = img_array[y1_clip, x1_clip]

    # Calculate the weights for interpolation
    # wa = (x0 + 1 - src_x) * (y0 + 1 - src_y)
    # wb = (x0 + 1 - src_x) * (src_y - y0)
    # wc = (src_x - x0) * (y0 + 1 - src_y)
    # wd = (src_x - x0) * (src_y - y0)

    # Expand dimensions of weights to match the input image shape
    # wa = wa[:, :, None]
    # wb = wb[:, :, None]
    # wc = wc[:, :, None]
    # wd = wd[:, :, None]

    # Compute the output image using the weights
    # output = wa * Ia + wb * Ib + wc * Ic + wd * Id



    output =  img_array[np.clip(src_y, 0, img_array.shape[0] - 1), np.clip(src_x, 0, img_array.shape[1] - 1)] +  \
              img_array[np.clip(src_y + 1, 0, img_array.shape[0] - 1), np.clip(src_x, 0, img_array.shape[1] - 1)] + \
            img_array[np.clip(src_y, 0, img_array.shape[0] - 1), np.clip(src_x + 1, 0, img_array.shape[1] - 1)] + \
            img_array[np.clip(src_y + 1, 0, img_array.shape[0] - 1), np.clip(src_x + 1, 0, img_array.shape[1] - 1)]

    # Handle borders
    if borderMode == cv2.BORDER_REPLICATE:
        mask = (src_x < 0) | (src_x >= img_array.shape[1]) | (src_y < 0) | (src_y >= img_array.shape[0])
        output[mask] = borderValue

    return output.astype(img_array.dtype)





def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3):
    """
    Construct neighbor graphs, including feature graph and spatial graph.
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
        n_neighbors = 6
        # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

    return data


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="correlation",
                               include_self=False):
    """Constructing feature neighbor graph according to expresss profiles"""

    feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)
    feature_graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    # print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""

    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)

    adj_spatial_omics1 = adj_spatial_omics1.toarray()  # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()

    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2 > 1, 1, adj_spatial_omics2)

    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1)  # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)

    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())

    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1 > 1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2 > 1, 1, adj_feature_omics2)

    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1)  # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)

    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }

    return adj


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    # X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    # adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def fix_seed(seed):
    # seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'