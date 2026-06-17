#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32gb
#PBS -l walltime=48:00:00
#PBS -N output_runall

cd $PBS_O_WORKDIR

# Always start clean and activate env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wip

echo "Started $PWD at $(date)"

# experiment name is defined as the name of the folder in which the script resides
EXP_NAME=$(basename "$PWD")

# Feature type options: "ECIF" or "PLEC"
FEATURE_TYPE="ECIF"

# Model type options: "xgb", "rf", "svr"
MODEL_TYPE="xgb"

# merge_dataset.py creates a processed folder where data is formatted for training
# data_name should be changed to reflect the file setup within exp_name/data/ and must contain 5 folds
python3 ../src/models/tree-regression-models/merge_dataset.py \
    --exp_name "$EXP_NAME" \
    --data_name "MCL1_cluster_fold-" \
    --features "$FEATURE_PATH"

# train_models.py takes in processed data and trains the model
# saves 5 cross validation models (one for each fold) to models_cv
# saves best params determined by optuna to exp_name/json/train_best_params.json
# json files for logging saved in exp_name/json/
# figures for optuna saved in exp_name/figures/
# saves 10 different seed models to exp_name/models_seeded/
python3 ../src/models/tree-regression-models/train_models.py \
    --exp_name "$EXP_NAME" \
    --model "$MODEL_TYPE" \
    --params "$BEST_PARAMS" \
    --feature_type "$FEATURE_TYPE"

# evaluates the 5 cross validation models using fep_{feature_type}_features.csv
    # saves to: exp_name/tree-regression_example_cv_predictions.csv
# evaluates the 10 seeded models using fep_{feature_type}_features.csv
    # saves to: exp_name/tree-regression_example_seeded_predictions.csv
python3 ../src/models/tree-regression-models/eval_script.py \
    --exp_name "$EXP_NAME" \
    --model "$MODEL_TYPE" \
    --feature_type "$FEATURE_TYPE"

echo "Finished at $(date)"