import random

import numpy as np
import torch


def read_test_walks():
    with open('walks.npy', 'rb') as f:
        walks = np.load(f)

    num_nodes = walks.max() + 1
    n_walks = walks.shape[0] // num_nodes

    assert n_walks * num_nodes == walks.shape[0]

    return walks, num_nodes


def generate_walks(
    num_nodes, num_walks, walk_length,
):
    walks = np.random.randint(0, num_nodes, (num_nodes * num_walks, walk_length))
    walks[:num_nodes, 0] = np.arange(num_nodes)
    return walks


def generate_edge_index(num_nodes, num_edges):
    src_lst = np.random.randint(0, num_nodes, num_edges)
    dst_lst = np.random.randint(0, num_nodes, num_edges)
    return torch.Tensor([src_lst, dst_lst]).to(torch.long)


def get_edge_index(walks):
    n_walks, walk_length = walks.shape
    
    edges = []
    for n_walk, step in np.ndindex(n_walks, walk_length - 1):
        edge = walks[n_walk, step:step+2]
        edges.append(edge)

    return torch.Tensor(edges).T.to(torch.long)


def add_loops(edge_index, num_nodes):
    edges = [edge.tolist() for edge in edge_index.T]
    for v in range(num_nodes):
        edges.append((v, v))
    return torch.Tensor(edges).T.to(torch.long)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)