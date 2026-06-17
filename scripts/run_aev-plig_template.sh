#!/bin/bash
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=1:00:00
#PBS -N output

cd $PBS_O_WORKDIR

# start clean and activate env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate aev-plig-manual
#conda activate ts_aev-plig

# make logs directory
mkdir -p logs

# define variables
EXP_NAME=$(basename "$PWD")
TRAIN_DATA=data/MCL1_cluster-fold_1_train.csv
VALID_DATA=data/MCL1_cluster-fold_1_valid.csv
# if combined does not exist
head -n 1 $TRAIN_DATA > data/MCL1_all.csv
tail -n +2 $TRAIN_DATA >> data/MCL1_all.csv
tail -n +2 $VALID_DATA >> data/MCL1_all.csv
DATA=data/MCL1_all.csv

echo "Started at $(date)"

python3 ../../src/models/AEV-PLIG-models/generate_graphs.py \
    --exp_name $EXP_NAME \
    --data $DATA \
    > logs/graphs.log 2>&1

python3 ../../src/models/AEV-PLIG-models/create_pytorch_data.py \
    --exp_name $EXP_NAME \
    --train_data $TRAIN_DATA \
    --valid_data $VALID_DATA \
    > logs/data.log 2>&1

python3 ../../train.py \
    --exp_name $EXP_NAME \
    --hidden_dim 32 \
    --head 1 \
    --gnn_layers 1 \
    > logs/train.log 2>&1

python3 ../../fep_predictions.py \
    --exp_name $EXP_NAME \
    --hidden_dim 32 \
    --head 1 \
    --gnn_layers 1 \
    > logs/predictions.log 2>&1

# ------------------------------
# # TRAINING COMMAND TEMPLATE (default args)
# ------------------------------
# python3 train.py \
#   --exp_name my_experiment \
#   --model GATv2Net \
#   --batch_size 128 \
#   --epochs 200 \
#   --hidden_dim 256 \
#   --head 3 \
#   --lr 0.00012291937615434127 \
#   --activation_function leaky_relu \
#   --gnn_layers 5

# ------------------------------
# # PREDICTIONS COMMAND TEMPLATE (default args)
# ------------------------------
# python3 train.py \
#   --exp_name my_experiment \
#   --hidden_dim 256 \
#   --head 3 \
#   --activation_function leaky_relu \
#   --gnn_layers 5


echo "Finished at $(date)"