#!/usr/bin/env bash

export TIME=0-24:0
export GPUS=2
export CPUS=64
export CONSTR=type_e_debug

export REPEAT=3
export MAX_JOBS=2

export script=$1
export name="${script#scripts/}"
export name="${name%.*}"

export WANDB_MODE=offline

sbatch --export=ALL --job-name=$name --time=$TIME\
       --gpus=$GPUS --cpus-per-task=$CPUS --constraint=$CONSTR\
       scripts/run_grid.sbatch

# Example: ./run_grid.sh scripts/ARRWPE/SVD/run_photo_dim.sh
