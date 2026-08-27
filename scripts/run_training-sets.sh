#!/bin/bash
#PBS -l select=1:ncpus=4:mem=256gb
#PBS -l walltime=48:00:00
#PBS -N out_trainingsets

cd $PBS_O_WORKDIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate wip

echo "Started at $(date)"

python3 ../src/training-sets/gen_cluster_trainingsets.py

echo "Finished at $(date)"