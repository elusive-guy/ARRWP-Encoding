#!/usr/bin/env bash

export TIME=0-24:0
export GPUS=1
export CPUS=32

export REPEAT=10

export config=$1
export name="${config#configs/}"
export name="${name%.*}"

export WANDB_MODE=offline

sbatch --export=ALL --job-name=$name --time=$TIME --gpus=$GPUS\
       --cpus-per-task=$CPUS --constraint=$CONSTR\
       scripts/run_config.sbatch $config $REPEAT

# Example: ./run_config.sh configs/ARRWPE/photo.yaml
