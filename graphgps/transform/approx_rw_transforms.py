import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import random_projection

from numba import jit, prange

from graphgps.transform.approx_rw_utils\
    import (remove_extra_loops,
            check_loops,
            count_walks,
            get_relations_for_stats,
            get_relations,
            add_self_relations_for_stats,
            add_self_relations,
            get_edge_idx,
            edge_to_id,
            id_to_edge)


@torch.no_grad()
def calculate_arrwpe_stats(
    walks, num_nodes, window_size=None,
    scale=True, edge_index=None,
):
    """
    Calculates statistics for ARRWP Encoding.

    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        num_nodes: int
            The number of nodes in the graph.
        window_size: int
            The size of the window for more effective
            walks using.
            If None, the walk length is used.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.
        edge_index: torch.Tensor
            An array of graph edges of size (2, edge_count).
            Indicates which pairs of nodes to consider.
            If None, all pairs are considered.
    
    Returns:
        abs_enc: torch.Tensor
            Absolute encoding for nodes.
            Equivalent to matrix ARWSE.
        rel_enc_index: torch.Tensor
            Source and Destination nodes for relative encoding.
        rel_enc_val: torch.Tensor
            Relative encoding values.
    """

    if window_size is None:
        window_size = walks.shape[1]

    arrwp = calculate_arrwp_matrix_for_stats(
        walks, num_nodes, window_size, 
        scale=scale, edge_index=edge_index.cpu().detach().numpy(),
    )
    rel_enc_index = arrwp.indices()
    rel_enc_val = arrwp.values()

    assert check_loops(
        rel_enc_index, num_nodes,
    )  # necessary condition for abs_enc and remove_extra_loops

    loop_mask = (rel_enc_index[0] == rel_enc_index[1])
    abs_enc = rel_enc_val[loop_mask]

    # remove extra loops
    rel_enc_index, rel_enc_val = remove_extra_loops(
        rel_enc_index, rel_enc_val, edge_index,
    )

    return abs_enc, rel_enc_index, rel_enc_val


@torch.no_grad()
def calculate_arrwp_matrix_for_stats(
    walks, num_nodes, window_size=None, 
    scale=True, edge_index=None,
):
    """
    Calculates the ARRWP matrix considering only pairs of
    nodes connected by edges from edge_index.

    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        num_nodes: int
            The number of nodes in the graph.
        window_size: int
            The size of the window for more effective
            walks using.
            If None, the walk length is used.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.
        edge_index: np.ndarray
            An array of graph edges of size (2, edge_count).
            Indicates which pairs of nodes to consider.
            If None, all pairs are considered.

    Returns:
        The ARRWP matrix of size (num_nodes, num_nodes,
        window_size) as sparse COO tensor.
    """

    if window_size is None:
        window_size = walks.shape[1]

    idx, values = calculate_arrwp_matrix_for_stats_(
        walks, num_nodes, window_size, 
        scale=scale, edge_index_=edge_index,
    )
    size = (num_nodes, num_nodes, window_size)

    return torch.sparse_coo_tensor(
        idx, values, size,
    ).coalesce()


@jit(nopython=True, parallel=True)
def calculate_arrwp_matrix_for_stats_(
    walks, num_nodes, window_size,
    scale=True, edge_index_=None,
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
    
    if edge_index_ is None:
        edge_index = get_relations_for_stats(walks, window_size)
    else:
        edge_index = edge_index_.astype(np.int64)
        edge_index = add_self_relations_for_stats(edge_index, num_nodes)

    num_edges = edge_index.shape[1]
    edge_ids_ = np.array([edge_to_id(edge, num_nodes)\
                            for edge in edge_index.T])
    edge_ids = np.sort(edge_ids_)

    arrwp = np.zeros((num_edges + 1, window_size))
    for depth in prange(window_size):
        for walk_id, step in np.ndindex(num_walks, num_steps):
            src = walks[walk_id, step]
            dst = walks[walk_id, step + depth]
            edge = (src, dst)

            idx = get_edge_idx(edge_ids, edge, num_edges, num_nodes)
            arrwp[idx, depth] += addition[src]

    idx_, values_ = [(0, 0)], []
    idx_.clear()

    for edge_idx in range(num_edges):
        edge_id = edge_ids[edge_idx]
        src, dst, _ = id_to_edge(edge_id, num_nodes)

        key = (src, dst)
        value = arrwp[edge_idx]

        idx_.append(key)
        values_.append(list(value))

    idx = np.array(idx_).T.astype(np.int64)
    values = np.array(values_)\
               .astype(np.float32)

    return idx, values


@torch.no_grad()
def calculate_arrwpe_proj_stats(
    walks, num_nodes, dim_reduction, dim_reduced,
    window_size=None, scale=True, edge_index=None,
):
    """
    Calculates the ARRWP matrix and projects it using
    dimensionality reduction method specified in dim_reduction.

    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        num_nodes: int
            The number of nodes in the graph.
        dim_reduction: str
            The name of the method used to reduce dimensionality.
        dim_reduced: int
            Dimension of the projected space.
        window_size: int
            The size of the window for more effective
            walks using.
            If None, the walk length is used.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.
        edge_index: np.ndarray
            An array of graph edges of size (2, edge_count).
            Indicates which pairs of nodes to consider.
            If None, all pairs are considered.

    Returns:
        The projected ARRWP matrix of size (num_nodes,
        dim_reduced) as a tensor or a sparse COO tensor
        depending on the dimensionality reduction method.
    """
    
    if window_size is None:
        window_size = walks.shape[1]

    idx, values = calculate_arrwp_matrix_(
        walks, num_nodes, window_size, 
        scale=scale, edge_index_=edge_index.cpu().detach().numpy(),
    )

    idx1 = idx[1]*window_size + idx[2]
    idx[1] = idx1
    idx = idx[:2]

    size = (num_nodes, num_nodes*window_size)

    if dim_reduction == 'LinearProjection':
        # expected to be projected by the encoder
        return torch.sparse_coo_tensor(
            idx, values, size,
        ).coalesce()

    if dim_reduction.endswith('Projection'):
        arrwp = csr_matrix(
            (values, idx),
            shape=size,
        )

        if dim_reduction == 'RandomProjection':
            transformer = random_projection.GaussianRandomProjection(dim_reduced)
            arrwp_proj = transformer.fit_transform(arrwp)
            return torch.Tensor(arrwp_proj)

        elif dim_reduction == 'SparseRandomProjection':
            # transformer = random_projection.SparseRandomProjection(
            #     dim_reduced, dense_output=True,
            # )
            # arrwp_proj = transformer.fit_transform(arrwp)
            # return torch.Tensor(arrwp_proj)

            transformer = random_projection.SparseRandomProjection(
                dim_reduced,
            )
            arrwp_proj = transformer.fit_transform(arrwp).tocoo()
            return torch.sparse_coo_tensor(
                np.array((arrwp_proj.row, arrwp_proj.col)), arrwp_proj.data,
                arrwp_proj.shape,
            )

    else:
        arrwp = torch.sparse_coo_tensor(
            idx, values, size,
        ).coalesce()
        
        if dim_reduction == 'TruncatedSVD':
            U, S, _ = torch.svd_lowrank(arrwp, q=dim_reduced)
            return U * S

        elif dim_reduction == 'PCA':
            U, S, _ = torch.pca_lowrank(arrwp, q=dim_reduced)
            return U * S

    raise ValueError(f"Unexpected dimensionality reduction algorithm: "\
                     f"{dim_reduction}")


@torch.no_grad()
def calculate_arrwp_matrix(
    walks, num_nodes, window_size=None, 
    scale=True, edge_index=None,
):
    """
    Calculates the ARRWP matrix considering only pairs of
    nodes connected by edges from edge_index.

    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        num_nodes: int
            The number of nodes in the graph.
        window_size: int
            The size of the window for more effective
            walks using.
            If None, the walk length is used.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.
        edge_index: np.ndarray
            An array of graph edges of size (2, edge_count).
            Indicates which pairs of nodes to consider.
            If None, all pairs are considered.

    Returns:
        The ARRWP matrix of size (num_nodes, num_nodes,
        window_size) as a sparse COO tensor.
    """

    if window_size is None:
        window_size = walks.shape[1]

    idx, values = calculate_arrwp_matrix_(
        walks, num_nodes, window_size, 
        scale=scale, edge_index_=edge_index,
    )
    size = (num_nodes, num_nodes, window_size)

    return torch.sparse_coo_tensor(
        idx, values, size,
    ).coalesce()


@torch.no_grad()
def calculate_arwpe_matrix(
    walks, num_nodes, window_size=None, scale=True,
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
            If None, the walk length is used.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.

    Returns:
        The ARWPE matrix of size (num_nodes, window_size).
    """

    if window_size is None:
        window_size = walks.shape[1]

    arwpe = calculate_arwpe_matrix_(
        walks, num_nodes, window_size, scale=scale,
    )
    return torch.Tensor(arwpe)


@torch.no_grad()
def calculate_arwse_matrix(
    walks, num_nodes, window_size=None, scale=True,
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
            If None, the walk length is used.
        scale: bool
            IF True, probabilities will be computed.
            If False, hits will be computed.

    Returns:
        The ARWSE matrix of size (num_nodes, window_size).
    """

    if window_size is None:
        window_size = walks.shape[1]
    
    arwse = calculate_arwse_matrix_(
        walks, num_nodes, window_size, scale=scale,
    )
    return torch.Tensor(arwse)


@jit(nopython=True)
def calculate_arrwp_matrix_(
    walks, num_nodes, window_size,
    scale=True, edge_index_=None,
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
    
    if edge_index_ is None:
        edge_index = get_relations(walks, window_size)
    else:
        edge_index = edge_index_.astype(np.int64)
        edge_index = add_self_relations(edge_index, num_nodes, window_size)

    num_edges = edge_index.shape[1]
    edge_ids_ = np.array([edge_to_id(edge, num_nodes)\
                            for edge in edge_index.T])
    edge_ids = np.sort(edge_ids_)

    arrwp = np.zeros(num_edges + 1)
    for walk_id, step in np.ndindex(num_walks, num_steps):
        for depth in range(window_size):
            src = walks[walk_id, step]
            dst = walks[walk_id, step + depth]
            edge = (src, dst, depth)

            idx = get_edge_idx(edge_ids, edge, num_edges, num_nodes)
            arrwp[idx] += addition[src]

    idx_, values_ = [(0, 0, 0)], []
    idx_.clear()

    for edge_idx in range(num_edges):
        edge_id = edge_ids[edge_idx]
        src, dst, depth = id_to_edge(edge_id, num_nodes)

        key = (src, dst, depth)
        value = arrwp[edge_idx]

        idx_.append(key)
        values_.append(value)

    idx = np.array(idx_).T.astype(np.int64)
    values = np.array(values_)\
               .astype(np.float32)

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

    self_walks = np.zeros((num_nodes, window_size))
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