#!/usr/bin/env bash

export TIME=0-6:0
export GPUS=1
export CPUS=32

export REPEAT=10

export config=$1
export name="${config#configs/}"
export name="${name%.*}"

export WANDB_MODE=offline

n=$(($REPEAT - 1))

for i in $(seq 0 $n);
do
    sbatch --export=ALL --job-name=$name --time=$TIME --gpus=$GPUS\
    --cpus-per-task=$CPUS --constraint=$CONSTR\
    scripts/run_pconfig.sbatch $config $i
done

# Example: ./run_pconfig.sh configs/ARRWPE/photo.yaml
