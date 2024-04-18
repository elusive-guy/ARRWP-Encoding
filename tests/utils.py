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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)