import torch
import numpy as np

from numba import jit


def remove_extra_loops(
    sur_index, sur_val, edge_index,
    check_condition=False, num_nodes=None,
):
    """
    Removes loops that do not occur in a graph.

    Args:
        sur_index: torch.Tensor
            A tensor of size (2, num_sur_edges) with extra
            loops. It must contain all loops which occur in
            ascending order.
        sur_val: torch.Tensor
            A tensor of size (num_sur_edges, val_dim) with
            values corresponding to sur_index.
        edge_index: torch.Tensor
            A tensor of size (2, num_edges) with edges in the
            graph.
        check_condition: bool
            Indicates whether to check sur_index conditions
            or not.
        num_nodes: int
            Number of nodes in the graph. Needed to check
            the condition.
    
    Returns:
        Edges and values with removed extra loops.
    """

    if check_condition:
        assert check_loops(sur_index, num_nodes)

    sur_loop_mask = (sur_index[0] == sur_index[1])
    index1 = sur_index[:, ~sur_loop_mask]
    val1 = sur_val[~sur_loop_mask]

    loop_mask = (edge_index[0] == edge_index[1])
    loops = torch.unique(edge_index[0][loop_mask])

    index2 = sur_index[:, sur_loop_mask][:, loops]
    val2 = sur_val[sur_loop_mask][loops]

    final_index = torch.cat([index1, index2], dim=1)
    final_val = torch.cat([val1, val2], dim=0)
    return final_index, final_val


def check_loops(edge_index, num_nodes):
    """
    Checks if all loops exist and are sorted.

    Args:
        edge_index: torch.Tensor
            A tensor of size (2, num_edges) with edges in a
            graph.
        num_nodes: int
            Number of nodes in the graph.

    Returns:
        Whether the condition completed or not.
    """

    loop_mask = (edge_index[0] == edge_index[1])
    loops = edge_index[0][loop_mask]
    seq = torch.arange(num_nodes).to(loops)
    return loops.shape == seq.shape and (loops == seq).all()


@jit(nopython=True)
def count_walks(walks, num_nodes, window_size, replace_zeros=False):
    """
    For each node, counts the number of walks that start from it.
    
    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        num_nodes: int
            The number of nodes in the graph.
        window_size: int
            The size of the window for more effective
            walks using.
        replace_zeros: bool
            If True, replace zero values in the resulting
            array with ones.

    Returns:
        An array with counted starting walks for each node.
    """

    walks_ = walks[..., :-window_size + 1].flatten()
    walks_count_ = np.bincount(walks_)
    
    walks_count = np.zeros(num_nodes)
    walks_count[:len(walks_count_)] = walks_count_

    if not walks_count.all() and replace_zeros:
        walks_count = np.where(walks_count, walks_count, 1)

    return walks_count


@jit(nopython=True)
def get_relations(walks, window_size):
    """
    Collects unique pairs of nodes which have at least one
    walk between them.

    Args:
        walks: np.ndarray
            2D array with walks in a graph.
        window_size: int
            The size of the window for more effective
            walks using.

    Returns:
        An array of size (2, relation_count) with unique
        node pairs.
    """

    num_walks = walks.shape[0]
    num_steps = walks.shape[1] - window_size + 1

    relations = set()
    for walk_id, step in np.ndindex(num_walks, num_steps):
        for depth in range(window_size):
            src = walks[walk_id, step]
            dst = walks[walk_id, step + depth]
            relations.add((src, dst))
    relations = list(relations)

    return np.array(relations, dtype=np.int64).T


@jit(nopython=True)
def add_self_relations(relations, num_nodes):
    """
    Adds a relation with itself for each node.
    
    Args:
        relations: np.ndarray
            2D array with relations between nodes.
        num_nodes: int
            The number of nodes in the graph.

    Returns:
        Updated relations of size (2, new_relation_count)
        with unique node pairs.
    """

    upd_relations = set()
    for src, dst in relations.T:
        upd_relations.add((src, dst))
    
    for v in range(num_nodes):
        upd_relations.add((v, v))

    upd_relations = list(upd_relations)
    return np.array(upd_relations).T
    

@jit(nopython=True)
def edge_to_id(edge, num_nodes):
    return edge[0]*num_nodes + edge[1]


@jit(nopython=True)
def id_to_edge(edge_id, num_nodes): 
    return edge_id // num_nodes, edge_id % num_nodes