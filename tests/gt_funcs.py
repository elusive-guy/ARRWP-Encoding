import numpy as np


def gt_count_walks(walks, num_nodes, window_size):
    walks_count = np.zeros(num_nodes)
    for walk in walks:
        for s in walk[:-window_size + 1]:
            walks_count[s] += 1
    return walks_count


def gt_calculate_arrwp_matrix(
    walks, num_nodes, window_size, scale=True,
):
    walk_length = walks.shape[-1]
    mx = np.zeros((num_nodes, num_nodes, window_size))
    for walk in walks:
        for i in range(walk_length - window_size + 1):
            s = walk[i]
            for k, v in enumerate(walk[i:i + window_size]):
                mx[s, v, k] += 1
    
    if scale:
        mx /= np.sum(mx, axis=1, keepdims=True)
    
    return mx


def gt_calculate_arwpe_matrix(walks, num_nodes, window_size, scale=True):
    mx = gt_calculate_arrwp_matrix(walks, num_nodes, window_size, scale)
    mx = mx.sum(0) - mx.diagonal().T
    return mx


def gt_calculate_arwse_matrix(walks, num_nodes, window_size, scale=True):
    mx = gt_calculate_arrwp_matrix(walks, num_nodes, window_size, scale)
    mx = mx.diagonal().T
    return mx