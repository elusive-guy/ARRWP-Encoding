import torch
import numpy as np

from numba import jit, prange

from graphgps.transform.approx_rw_utils import count_walks


def calculate_arrwp_matrix(
    walks, num_nodes, window_size, scale=True,
):
    """
    Calculates the ARRWP matrix.

    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        num_nodes: int
            The number of nodes in the graph.
        window_size: int
            The size of the window for more effective
            walks using.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.

    Returns:
        The ARRWP matrix with size (num_nodes, num_nodes,
        window_size) as sparse COO tensor.
    """

    idx, values = calculate_arrwp_matrix_(
        walks, num_nodes, window_size, scale,
    )
    size = (num_nodes, num_nodes, window_size)
    
    return torch.sparse_coo_tensor(idx, values, size)


def calculate_arwpe_matrix(
    walks, num_nodes, window_size, scale=True,
):
    """
    Calculates the ARWPE matrix.

    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        num_nodes: int
            The number of nodes in the graph.
        window_size: int
            The size of the window for more effective walks using.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.

    Returns:
        The ARWPE matrix with size (num_nodes, window_size).
    """

    arwpe = calculate_arwpe_matrix_(
        walks, num_nodes, window_size, scale,
    )
    return torch.Tensor(arwpe)


def calculate_arwse_matrix(
    walks, num_nodes, window_size, scale=True,
):
    """
    Calculates the ARWSE matrix.

    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        num_nodes: int
            The number of nodes in the graph.
        window_size: int
            The size of the window for more effective walks using.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.

    Returns:
        The ARWSE matrix with size (num_nodes, window_size).
    """
    
    arwse = calculate_arwse_matrix_(
        walks, num_nodes, window_size, scale,
    )
    return torch.Tensor(arwse)


@jit(nopython=True, parallel=True)
def calculate_arrwp_matrix_(
    walks, num_nodes, window_size, scale=True,
):
    num_walks = walks.shape[0]
    num_steps = walks.shape[1] - window_size + 1

    if scale:
        addition = 1 / count_walks(
                           walks, num_nodes,
                           window_size, replace_zeros=True,
                       )
    else:
        addition = np.ones(num_nodes)

    idx_lst = [[(0, 0, 0)] for _ in range(window_size)]
    values_lst = [[0.0] for _ in range(window_size)]

    for depth in prange(window_size):
        idx_ = idx_lst[depth]
        values_ = values_lst[depth]
        for walk_id, step in np.ndindex(num_walks, num_steps):
            src = walks[walk_id, step]
            dst = walks[walk_id, step + depth]
            idx_.append((src, dst, depth - 1 + 1))
            values_.append(addition[src])

    def concatenate_lists(lst):
        n = sum([len(item) for item in lst])
        d = len(lst[0][0]) if isinstance(lst[0][0], tuple) else 1

        id = 0
        mx = np.empty((n, d))
        for item in lst:
            delta = len(item)
            arr = np.array(item)

            if len(arr.shape) == 1:
                mx[id:id + delta, 0] = arr
            else:
                mx[id:id + delta] = arr
            
            id += delta

        return mx
    
    idx = concatenate_lists(idx_lst).T.astype(np.int32)
    values = concatenate_lists(values_lst)[:, 0].astype(np.float32)

    return idx, values


@jit(nopython=True, parallel=True)
def calculate_arwpe_matrix_(
    walks, num_nodes, window_size, scale=True,
):
    num_walks = walks.shape[0]
    num_steps = walks.shape[1] - window_size + 1

    if scale:
        addition = 1 / count_walks(
                           walks, num_nodes, 
                           window_size, replace_zeros=True,
                       )
    else:
        addition = np.ones(num_nodes)

    arwpe = np.zeros((num_nodes, window_size))
    for depth in prange(window_size):
        for walk_id, step in np.ndindex(num_walks, num_steps):
            src = walks[walk_id, step]
            dst = walks[walk_id, step + depth]
            arwpe[dst, depth] += addition[src] * (dst != src)

    return arwpe.astype(np.float32)


@jit(nopython=True, parallel=True)
def calculate_arwse_matrix_(
    walks, num_nodes, window_size, scale=True,
):    
    num_walks = walks.shape[0]
    num_steps = walks.shape[1] - window_size + 1

    self_walks = np.zeros((num_nodes, window_size), dtype=np.int64)
    total_walks = count_walks(
        walks, num_nodes,
        window_size, replace_zeros=True,
    ).reshape(-1, 1)
    
    for depth in prange(window_size):
        for walk_id, step in np.ndindex(num_walks, num_steps):
            src = walks[walk_id, step]
            dst = walks[walk_id, step + depth]
            self_walks[dst, depth] += (src == dst)

    if scale:
        arwse = np.divide(self_walks, total_walks).astype(np.float32)
    else:
        arwse = self_walks.astype(np.float32)

    return arwse