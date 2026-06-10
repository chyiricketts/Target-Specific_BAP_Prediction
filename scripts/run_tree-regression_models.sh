#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32gb
#PBS -l walltime=48:00:00
#PBS -N output_runall

cd $PBS_O_WORKDIR

# Always start clean and activate env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wip

echo "Started $PWD at $(date)"

# define variables
# EXP_NAME=$(basename "$PWD")
FEATURE_PATH="../../ECIF_data/raw_all_ecif_features.csv"
# Feature type options: "ECIF" or "PLEC"
FEATURE_TYPE="ECIF"
# Model type options: "xgb", "rf", "svr"
MODEL_TYPE="xgb"

python3 ../src/models/tree-regression-models/merge_dataset.py \
    --exp_name "testing1" \
    --data_name "MCL1/cluster-split/MCL1_cluster_fold-" \
    --features "$FEATURE_PATH"

python3 ../../optuna_tune.py \
    --exp_name "$EXP_NAME" \
    --model "$MODEL_TYPE"

python3 ../../train_models.py \
    --exp_name "$EXP_NAME" \
    --model "$MODEL_TYPE" \
    --params "$BEST_PARAMS" \
    --feature_type "$FEATURE_TYPE"

python3 ../../eval_script.py \
    --exp_name "$EXP_NAME" \
    --model "$MODEL_TYPE" \
    --feature_type "$FEATURE_TYPE"

echo "Finished at $(date)"