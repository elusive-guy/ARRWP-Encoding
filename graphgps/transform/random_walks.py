from node2vec.model import Node2Vec


def simulate_random_walks(
    data, walk_length, n_walks,
    p, q, workers, verbose, rand_seed,
):
    """
    Simulates random walks using node2vec sampler.

    Args:
        data:
            Graph in PyG format.
        walk_length: int
            Length of walk per source node.
        n_walks: int
            Number of walks per source node.
        p: float
            Return hyperparameter.
        q: float
            Inout hyperparameter.
        workers: int
            Number of worker threads used.
        verbose: bool
            Verbosity of the output.
        rand_seed: int
            Seed for the random number generator. Use a fixed random seed to produce
            deterministic runs. To ensure fully deterministic runs you also need to
            set workers = 1.

    Returns:
        Data with random_walks attribute added.
    """

    # create node2vec sampler
    src, dst = data.edge_index.numpy()
    node2vec_model = Node2Vec(
        src, dst,
        graph_is_directed=data.is_directed(),
    )
    
    # simulate random walks
    node2vec_model.simulate_walks(
        walk_length=walk_length,
        n_walks=n_walks,
        p=p,
        q=q,
        workers=workers,
        verbose=verbose,
        rand_seed=rand_seed,
    )
    data.random_walks = node2vec_model.walks

    return data