#!/bin/bash
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -l walltime=24:00:00
#PBS -N output_features

cd $PBS_O_WORKDIR

# Always start clean
source ~/miniforge3/etc/profile.d/conda.sh

# Activate your environment
conda activate plec
echo "Started at $(date)"
python3 ../src/models/tree-regression-models/plec_features.py
# Works. takes 14 hours to run - 38 hours for Pdbbind + bindingnet
echo "Finished at $(date)"

# Activate your environment
conda activate ts_BAP
echo "Started at $(date)"
python3 ../src/models/tree-regression-models/ecif_features.py
echo "Finished at $(date)"