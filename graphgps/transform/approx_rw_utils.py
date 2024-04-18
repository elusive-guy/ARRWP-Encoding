import numpy as np

from numba import jit


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