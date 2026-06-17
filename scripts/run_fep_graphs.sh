#!/bin/bash
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=10:00:00
#PBS -N output_fep-graphs

cd $PBS_O_WORKDIR

# Always start clean
source ~/miniforge3/etc/profile.d/conda.sh

# Activate your environment
conda activate aev-plig-manual

echo "Started at $(date)"

# this is only for running once for fep graph generation
python3 ../src/models/AEV-PLIG-models/fep_graphs.py \

echo "Finished at $(date)"