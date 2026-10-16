"""Run the stLearn PSTS trajectory for the osteogenic, adipogenic and
chondrogenic lineages.

The stLearn pipeline -- Pham et al., *Robust mapping of spatiotemporal
trajectories and cell-cell interactions in healthy and diseased tissues*,
Nat. Commun. 14, 7739 (2023) -- is reimplemented here directly rather than
called as a library: DBSCAN localisation of subclusters, PAGA, diffusion
pseudotime, and the global pseudotime space, followed by a minimum spanning
arborescence over the subcluster graph. Subcluster edges are weighted by a
combination of expression distance and spatial distance, so the resulting
pseudotime follows the tissue rather than only the transcriptome.

Being a reimplementation, it runs on any expression matrix, which is what lets
the same code produce both the sensitivity-only arm (the stLearn baseline) and
the FOCUS arm of the comparison.

One trajectory is produced per lineage.
"""
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cdist
from natsort import natsorted
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

BASE = '/data_temp/mnt/raid5/qnf/bone_project/project_bone'
OUT = f'{BASE}/downstream/trajectory/output'
FIG = f'{OUT}/figures'

# Input of this step: the scored section subsampled to 50k cells with k-means
# clusters, written by 00_subsample.py.
IN_H5AD = f'{OUT}/nc_psts_trajectory.h5ad'

print("Loading data...")
adata_full = ad.read_h5ad(IN_H5AD)
print(f"Full shape: {adata_full.shape}")

# FUNCTIONS

def localization(adata, use_label, eps=300.0, min_samples=10):
    """DBSCAN spatial subclustering within each cluster."""
    if 'sub_cluster_labels' in adata.obs.columns:
        adata.obs = adata.obs.drop('sub_cluster_labels', axis=1)

    subclusters_list = []
    for i in adata.obs[use_label].unique():
        tmp = adata.obs[adata.obs[use_label] == i].copy()
        clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree').fit(
            tmp[['imagecol', 'imagerow']]
        )
        labels = clustering.labels_
        sublabels = [str(i) + '_' + str(l) for l in labels]
        tmp['sub_labels'] = sublabels
        subclusters_list.append(tmp['sub_labels'])

    subclusters = pd.concat(subclusters_list)
    adata.obs['sub_cluster_labels'] = subclusters

    converted = dict(enumerate(adata.obs['sub_cluster_labels'].unique()))
    inv_map = {v: str(k) for k, v in converted.items()}
    adata.obs['sub_cluster_labels'] = adata.obs['sub_cluster_labels'].replace(inv_map)
    adata.obs['sub_cluster_labels'] = pd.Categorical(
        values=np.array(adata.obs['sub_cluster_labels']).astype('U'),
        categories=natsorted(np.unique(np.array(adata.obs['sub_cluster_labels'])).astype('U'))
    )

    labels_cat = list(map(int, adata.obs[use_label].cat.categories))
    cat_ind = {labels_cat[i]: i for i in range(len(labels_cat))}
    adata.uns[use_label + '_index_dict'] = cat_ind

def build_split_node_and_centroids(adata, use_label, threshold_spots=20):
    """Build split_node dict and centroid_dict."""
    cat_inds = adata.uns[use_label + '_index_dict']
    split_node = {}
    for label in adata.obs[use_label].cat.categories:
        meaningful_sub = []
        for i in adata.obs[adata.obs[use_label] == label]['sub_cluster_labels'].unique():
            if len(adata.obs[adata.obs['sub_cluster_labels'] == str(i)]) > threshold_spots:
                meaningful_sub.append(i)
        idx = cat_inds[int(label)]
        split_node[str(idx)] = meaningful_sub

    adata.uns['split_node'] = split_node

    clf = NearestCentroid()
    valid_mask = adata.obs['sub_cluster_labels'].notna()
    clf.fit(adata.obs.loc[valid_mask, ['imagecol', 'imagerow']].values,
            adata.obs.loc[valid_mask, 'sub_cluster_labels'])
    centroid_dict = dict(zip(clf.classes_.astype(int), clf.centroids_, strict=False))

    for cl in adata.obs['sub_cluster_labels'].unique():
        cl_points = adata.obs[adata.obs['sub_cluster_labels'] == cl][['imagecol', 'imagerow']].values
        if len(cl_points) > 0:
            dists = np.sum((cl_points - centroid_dict[int(cl)])**2, axis=1)
            centroid_dict[int(cl)] = cl_points[np.argmin(dists)]

    adata.uns['centroid_dict'] = centroid_dict
    return split_node

def run_pseudotimespace_global(adata, use_label, list_clusters, w=0.0, use_rep='X_pca', n_dims=30):
    """Build minimum spanning arborescence trajectory graph."""
    cat_inds = adata.uns[use_label + '_index_dict']
    inds_cat = {v: k for k, v in cat_inds.items()}

    if isinstance(list_clusters[0], str):
        query_nodes = [cat_inds[int(label)] for label in list_clusters]
    else:
        query_nodes = list_clusters

    # Order by pseudotime
    mean_dpt = []
    for node in query_nodes:
        mean_dpt.append(
            adata.obs[adata.obs[use_label] == str(inds_cat[node])]['dpt_pseudotime'].median()
        )
    query_nodes = list(np.array(query_nodes)[np.argsort(mean_dpt)])

    print(f"  Trajectory path: {' -> '.join([str(inds_cat[n]) for n in query_nodes])}")

    split_node = adata.uns['split_node']
    centroid_dict = adata.uns['centroid_dict']
    centroid_dict = {int(k): v for k, v in centroid_dict.items()}

    # Build edge list between consecutive clusters' subclusters
    edge_list = []
    for i in range(len(query_nodes) - 1):
        subs_i = split_node[str(query_nodes[i])]
        subs_j = split_node[str(query_nodes[i + 1])]
        for s1 in subs_i:
            for s2 in subs_j:
                edge_list.append((int(s1), int(s2)))

    # Build directed graph
    H_sub = nx.DiGraph()
    for e in edge_list:
        H_sub.add_edge(e[0], e[1], weight=1.0)

    # Add virtual root
    for node in split_node[str(query_nodes[0])]:
        H_sub.add_edge(9999, int(node), weight=0.0)

    prepare_root = [centroid_dict[int(n)] for n in split_node[str(query_nodes[0])]]
    prepare_root = np.array(prepare_root)
    centroid_dict[9999] = prepare_root.mean(axis=0)

    # Compute edge weights
    query_dict = {}
    order_dict = {}
    for node_idx in query_nodes:
        for order, j in enumerate(
            adata.obs[adata.obs[use_label] == str(inds_cat[node_idx])]['sub_cluster_labels'].unique()
        ):
            query_dict[int(j)] = int(node_idx)
            order_dict[int(j)] = int(order)

    order_big_dict = {j: i for i, j in enumerate(query_nodes)}

    for i in range(len(query_nodes) - 1):
        # Gene expression distance
        c1 = str(inds_cat[query_nodes[i]])
        c2 = str(inds_cat[query_nodes[i + 1]])
        subs1 = [s for s in split_node[str(query_nodes[i])]]
        subs2 = [s for s in split_node[str(query_nodes[i + 1])]]

        # Spatial distance matrix
        coords1 = np.array([centroid_dict[int(s)] for s in subs1])
        coords2 = np.array([centroid_dict[int(s)] for s in subs2])
        sdm = cdist(coords1, coords2, 'euclidean')
        if sdm.max() > 0:
            sdm = sdm / sdm.max()

        # Gene expression distance (cosine on PCA)
        ge_dists = np.zeros((len(subs1), len(subs2)))
        for si, s1 in enumerate(subs1):
            pca1 = adata[adata.obs['sub_cluster_labels'] == str(s1)].obsm[use_rep][:, :n_dims]
            for sj, s2 in enumerate(subs2):
                pca2 = adata[adata.obs['sub_cluster_labels'] == str(s2)].obsm[use_rep][:, :n_dims]
                ge_dists[si, sj] = cdist(pca1, pca2, 'cosine').mean()
        if ge_dists.max() > 0:
            ge_dists = ge_dists / ge_dists.max()

        # Set weights on edges
        for si, s1 in enumerate(subs1):
            for sj, s2 in enumerate(subs2):
                weight = ge_dists[si, sj] * w + sdm[si, sj] * (1 - w)
                if H_sub.has_edge(int(s1), int(s2)):
                    H_sub[int(s1)][int(s2)]['weight'] = weight

    # Minimum spanning arborescence
    try:
        tree = nx.algorithms.tree.minimum_spanning_arborescence(H_sub)
    except nx.exception.NetworkXException as e:
        print(f"  MSA failed: {e}, using minimum spanning tree instead")
        tree = nx.minimum_spanning_tree(H_sub.to_undirected())
        tree = tree.to_directed()

    adata.uns['PTS_graph'] = {
        'graph': nx.to_scipy_sparse_array(tree),
        'node_dict': {i: n for i, n in enumerate(tree.nodes)},
    }

    return tree, centroid_dict, query_nodes

# Step 1: Prepare data - use existing clustering (kmeans)
print("\n--- Step 1: Prepare clustering ---")

# Convert kmeans to proper categorical
adata_full.obs['kmeans'] = pd.Categorical(adata_full.obs['kmeans'].astype(str))
use_label = 'kmeans'

# Check cluster scores to determine trajectory paths
cluster_scores = adata_full.obs.groupby(use_label)[
    ['LepR_MSC_score', 'Osteoblast_score', 'Osteocyte_score',
     'Adipocyte_score', 'Chondrocyte_score']
].mean()
print("Cluster mean scores:")
print(cluster_scores.round(3).to_string())

cluster_spatial = adata_full.obs.groupby(use_label)[['cx', 'cy']].mean()
print("\nCluster spatial centroids (cx=bone side low, marrow side high):")
print(cluster_spatial.round(0).to_string())

# Identify key clusters
msc_cluster = cluster_scores['LepR_MSC_score'].idxmax()
osteo_cluster = cluster_scores['Osteocyte_score'].idxmax()
adipo_cluster = cluster_scores['Adipocyte_score'].idxmax()
chondro_cluster = cluster_scores['Chondrocyte_score'].idxmax()

# Also find osteoblast-intermediate cluster
ob_cluster = cluster_scores['Osteoblast_score'].idxmax()

print(f"\nKey clusters:")
print(f"  MSC start: {msc_cluster} (score={cluster_scores.loc[msc_cluster, 'LepR_MSC_score']:.3f})")
print(f"  Osteoblast: {ob_cluster} (score={cluster_scores.loc[ob_cluster, 'Osteoblast_score']:.3f})")
print(f"  Osteocyte end: {osteo_cluster} (score={cluster_scores.loc[osteo_cluster, 'Osteocyte_score']:.3f})")
print(f"  Adipocyte end: {adipo_cluster} (score={cluster_scores.loc[adipo_cluster, 'Adipocyte_score']:.3f})")
print(f"  Chondrocyte end: {chondro_cluster} (score={cluster_scores.loc[chondro_cluster, 'Chondrocyte_score']:.3f})")

# Step 2: Run DBSCAN localization
print("\n--- Step 2: DBSCAN localization ---")

# Spatial range is ~12000 x 6000 pixels
# eps=300 should give reasonable subclusters (not too many, not too few)
localization(adata_full, use_label=use_label, eps=400, min_samples=30)

# Count subclusters per cluster
for label in sorted(adata_full.obs[use_label].unique(), key=int):
    subs = adata_full.obs[adata_full.obs[use_label] == label]['sub_cluster_labels'].unique()
    n_meaningful = sum(1 for s in subs
                       if len(adata_full.obs[adata_full.obs['sub_cluster_labels'] == s]) > 20)
    print(f"  Cluster {label}: {n_meaningful} subclusters (of {len(subs)} total)")

# Build split_node and centroids
build_split_node_and_centroids(adata_full, use_label, threshold_spots=30)
print(f"  Total meaningful subclusters: {sum(len(v) for v in adata_full.uns['split_node'].values())}")

# Step 3: Run PAGA + DPT
print("\n--- Step 3: PAGA + DPT ---")

# Recompute neighbors
sc.pp.neighbors(adata_full, n_neighbors=15, use_rep='X_pca', n_pcs=30)
sc.tl.paga(adata_full, groups=use_label)

# Set root for DPT: use cell with highest MSC score
msc_cells = adata_full.obs[adata_full.obs[use_label] == msc_cluster]
root_cell_idx = msc_cells['LepR_MSC_score'].idxmax()
root_cell_pos = list(adata_full.obs.index).index(root_cell_idx)
adata_full.uns['iroot'] = root_cell_pos
print(f"  Root cell: {root_cell_idx} (MSC score={msc_cells.loc[root_cell_idx, 'LepR_MSC_score']:.3f})")

sc.tl.diffmap(adata_full)
sc.tl.dpt(adata_full)
print(f"  DPT range: [{adata_full.obs['dpt_pseudotime'].min():.3f}, {adata_full.obs['dpt_pseudotime'].max():.3f}]")

# Step 4: Run trajectory for each lineage
print("\n--- Step 4: Build trajectories ---")

# Define paths for each lineage
# Need to find intermediate clusters along each path

# For osteogenic: MSC → OB-enriched → Osteocyte
# For adipogenic: MSC → intermediate → Adipocyte
# For chondrogenic: MSC → intermediate → Chondrocyte

# Sort clusters by DPT to find natural ordering
cluster_dpt = adata_full.obs.groupby(use_label)['dpt_pseudotime'].median()
print("\nCluster median DPT:")
print(cluster_dpt.sort_values().to_string())

# Find path clusters using PAGA connectivity + scores
paga_conn = adata_full.uns['paga']['connectivities'].toarray()
cluster_labels = list(adata_full.obs[use_label].cat.categories)

def find_trajectory_path(start, end, scores_col, n_intermediate=2):
    """Find path from start to end cluster using PAGA + score gradient."""
    start_idx = cluster_labels.index(start)
    end_idx = cluster_labels.index(end)

    # Use PAGA graph to find connected path
    G = nx.from_numpy_array(paga_conn)
    G_labels = {i: cluster_labels[i] for i in range(len(cluster_labels))}

    # Weight edges by inverse connectivity
    for u, v, d in G.edges(data=True):
        if d['weight'] > 0:
            d['weight'] = 1.0 / d['weight']
        else:
            d['weight'] = 1000

    try:
        path_indices = nx.shortest_path(G, start_idx, end_idx, weight='weight')
        path = [cluster_labels[i] for i in path_indices]
    except nx.NetworkXNoPath:
        # Fallback: use DPT ordering
        all_clusters_sorted = cluster_dpt.sort_values().index.tolist()
        start_pos = all_clusters_sorted.index(start)
        end_pos = all_clusters_sorted.index(end)
        if start_pos < end_pos:
            path = all_clusters_sorted[start_pos:end_pos+1]
        else:
            path = all_clusters_sorted[end_pos:start_pos+1][::-1]

    # If path is too long, keep start, end, and best intermediates
    if len(path) > 5:
        intermediates = path[1:-1]
        # Score intermediates by their relevant score
        inter_scores = [(c, cluster_scores.loc[c, scores_col]) for c in intermediates]
        inter_scores.sort(key=lambda x: x[1], reverse=True)
        best_inter = [x[0] for x in inter_scores[:n_intermediate]]
        # Order by DPT
        best_inter.sort(key=lambda c: cluster_dpt[c])
        path = [start] + best_inter + [end]

    return path

# Osteogenic path
osteo_path = find_trajectory_path(msc_cluster, osteo_cluster, 'Osteoblast_score', n_intermediate=2)
print(f"\nOsteogenic path: {' → '.join(osteo_path)}")
for c in osteo_path:
    print(f"  Cluster {c}: MSC={cluster_scores.loc[c,'LepR_MSC_score']:.3f}, "
          f"OB={cluster_scores.loc[c,'Osteoblast_score']:.3f}, "
          f"OC={cluster_scores.loc[c,'Osteocyte_score']:.3f}, "
          f"cx={cluster_spatial.loc[c,'cx']:.0f}")

# Adipogenic path
adipo_path = find_trajectory_path(msc_cluster, adipo_cluster, 'Adipocyte_score', n_intermediate=1)
print(f"\nAdipogenic path: {' → '.join(adipo_path)}")
for c in adipo_path:
    print(f"  Cluster {c}: MSC={cluster_scores.loc[c,'LepR_MSC_score']:.3f}, "
          f"Adipo={cluster_scores.loc[c,'Adipocyte_score']:.3f}, "
          f"cx={cluster_spatial.loc[c,'cx']:.0f}")

# Chondrogenic path
chondro_path = find_trajectory_path(msc_cluster, chondro_cluster, 'Chondrocyte_score', n_intermediate=1)
print(f"\nChondrogenic path: {' → '.join(chondro_path)}")
for c in chondro_path:
    print(f"  Cluster {c}: MSC={cluster_scores.loc[c,'LepR_MSC_score']:.3f}, "
          f"Chondro={cluster_scores.loc[c,'Chondrocyte_score']:.3f}, "
          f"cx={cluster_spatial.loc[c,'cx']:.0f}")

# Run pseudotimespace_global for each
print("\n--- Building trajectory graphs ---")
trajectories = {}

for name, path in [('Osteogenic', osteo_path), ('Adipogenic', adipo_path), ('Chondrogenic', chondro_path)]:
    print(f"\n{name} trajectory:")
    try:
        tree, centroids, ordered_nodes = run_pseudotimespace_global(
            adata_full, use_label, path, w=0.3, use_rep='X_pca', n_dims=30
        )
        trajectories[name] = {
            'tree': tree,
            'centroids': centroids,
            'path': path,
            'ordered_nodes': ordered_nodes,
        }
        print(f"  Nodes: {tree.number_of_nodes()}, Edges: {tree.number_of_edges()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

# Step 5: Plot trajectory figure
print("\n--- Step 5: Plotting ---")

# Figure assembly is not part of this script. `trajectories` carries the trees,
# the subcluster centroids and the ordered node paths needed to draw them.
print("Trajectory computation complete. Plotting next...")
print(f"\nTrajectories computed: {list(trajectories.keys())}")
for name, traj in trajectories.items():
    print(f"  {name}: {traj['tree'].number_of_nodes()} nodes, path={traj['path']}")
