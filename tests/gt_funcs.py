import numpy as np


def gt_count_walks(walks, num_nodes, window_size):
    walks_count = np.zeros(num_nodes)
    for walk in walks:
        for s in walk[:-window_size + 1]:
            walks_count[s] += 1
    return walks_count


def gt_calculate_arrwp_matrix(
    walks, num_nodes, window_size, scale=True, edge_index=None,
    self_loops=False,
):
    walk_length = walks.shape[-1]
    mx = np.zeros((num_nodes, num_nodes, window_size))
    for walk in walks:
        for i in range(walk_length - window_size + 1):
            s = walk[i]
            for k, v in enumerate(walk[i:i + window_size]):
                mx[s, v, k] += 1
    
    if scale:
        denum_ = np.sum(mx, axis=1, keepdims=True)
        denum = np.where(denum_, denum_, 1)
        mx /= denum

    if edge_index is not None:
        edge_set = set(tuple(edge.tolist()) for edge in edge_index.T)
        for s, v in np.ndindex(num_nodes, num_nodes):
            if (s, v) not in edge_set and (not self_loops or s != v):
                mx[s, v] = 0
    
    return mx


def gt_calculate_arwpe_matrix(walks, num_nodes, window_size, scale=True):
    mx = gt_calculate_arrwp_matrix(walks, num_nodes, window_size, scale)
    mx = mx.sum(0) - mx.diagonal().T
    return mx


def gt_calculate_arwse_matrix(walks, num_nodes, window_size, scale=True):
    mx = gt_calculate_arrwp_matrix(walks, num_nodes, window_size, scale)
    mx = mx.diagonal().T
    return mx