# ARRWP Encoding

In this work we introduce approximated versions of RRWP (Relative Random Walk Probabilities), RWPE (Random Walk Positional Encoding), and RWSE (Random Walk Structural Encoding) matrices.
We called them as follows: ARRWP (Approximated RRWP), ARWPE (Approximated RWPE), and ARWSE (Approximated RWSE).
These matrices are obtained using the [node2vec](https://snap.stanford.edu/node2vec) sampler and can be applied to graphs with hundreds of millions nodes.
Although we were unable to improve the performance of Exphormer on small and large graphs, we believe that these matrices can be applied in other settings.

### Python environment setup with Conda

```bash
conda create -n arrwpe python=3.9
conda activate arrwpe

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install torchmetrics \
            performer-pytorch \
            ogb \
            tensorboardX \
            wandb \
            numba

pip install ./node2vec

pip uninstall torch-spline-conv

conda clean --all
```

### Running ARRWPE, ARWPE, and ARWSE

The scripts were written for the Slurm system.
As an example, you can execute the following command to use ARRWPE on ogbn-arxiv:

```bash
./run_config.sh configs/ARRWPE/arxiv.yaml
```

To make parallel runs with different seeds, you can use:

```bash
./run_pconfig.sh configs/ARRWPE/arxiv.yaml
```

To make grid search, you can use:

```bash
./run_grid.sh scripts/ARRWPE/SVD/run_arxiv_dim.sh
```

### Guide on config files

Most of the configs are shared with [Exphormer](https://github.com/hamed1375/Exphormer) code. You can change the following parameters in the config files for different variants of our approximations:

```
prep:
  random_walks:
    enable: True  # Set True for simulating random walks,
        # set False otherwise
    walk_length: 256  # Set the walk length
    n_walks: 100  # Set the number of walks
        # Please note that we simulate n_walks random
        # walks starting from each node
    p: 2.0  # If less than 1, we will more likely go backward
    q: 0.5  # If less than 1, we will more likely go forward 
    verbose: True  # Set True if you want to see the progress 
        # of walking, set False otherwise

posenc_ARRWPE:  # (similar setting for ARWPE and ARWSE)
  enable: True  # Set True if you want enable ARRWPE,
        # set False otherwise
  window_size: none  # This option can be used for more 
        # effective walks processing (we did not investigate
        # this feature)
  scale: True  # Set True if you want to calculate probabilities, 
        # otherwise you will get the number of hits from i to j
```