#!/bin/bash
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -l walltime=10:00:00
#PBS -N output_runall

cd $PBS_O_WORKDIR

# Always start clean and activate env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wip
PYTHONPATH=../../src \

echo "Started $PWD at $(date)"

# define variables
EXP_NAME=$(basename "$PWD")
# Feature type options: "ECIF" or "PLEC"
FEATURE_TYPE="ECIF"
# Model type options: "xgb", "rf", "svr"
MODEL_TYPE="xgb"

python3 ../../src/models/tree-regression-models/merge_dataset.py \
    --exp_name "$EXP_NAME" \
    --data_name "MCL1_cluster_fold-" \
    --features "$FEATURE_TYPE"

echo "Finished with merging"

python3 ../../src/models/tree-regression-models/train_models.py \
    --exp_name "$EXP_NAME" \
    --model "$MODEL_TYPE" \
    --params "$BEST_PARAMS" \
    --feature_type "$FEATURE_TYPE"

echo "Finished with training"

python3 ../../src/models/tree-regression-models/eval_script.py \
    --exp_name "$EXP_NAME" \
    --model "$MODEL_TYPE" \
    --feature_type "$FEATURE_TYPE"

echo "Finished at $(date)"