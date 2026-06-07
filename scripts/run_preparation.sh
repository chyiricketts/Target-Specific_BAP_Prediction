#!/bin/bash
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=10:00:00
#PBS -N out_trainingsets

cd $PBS_O_WORKDIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate wip

echo "Started at $(date)"

# Creates data/processed/all_protein.csv with local path to all used protein files in the 3 databases
# python3 ../src/data/all_proteins.py

# FEP Prediction info
python3 ../src/data/fep_predictioninfo.py

echo "Finished at $(date)"