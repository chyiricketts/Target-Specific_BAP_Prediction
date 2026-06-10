# to merge ECIF or PLEC features to the training dataset

import pandas as pd
import os
import numpy as np
import argparse
import json
from utils import ExperimentLogger

BASE_DIR = Path.cwd().parents[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    return parser.parse_args()


def merge_datasets(split_path, feature_df, outpath, logger, split_name, fold_idx):
    split_df = pd.read_csv(split_path)

    print(f"\n[{split_name}] split size: {split_df.shape}")
    print(f"[{split_name}] feature size: {feature_df.shape}")

    # merge
    merged = split_df.merge(feature_df, on="unique_id", how="inner")

    print(f"[{split_name}] merged size: {merged.shape}")

    # ensure output folder exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    merged.to_csv(outpath, index=False)

    # print a summary
    print(f"\n--------------Fold {fold_idx}----------------")
    print(f"Split shape: {list(split_df.shape)}")
    print(f"Feature_shape: {list(feature_df.shape)}")
    print(f"Merged_shape: {list(merged.shape)}")
    print(f"n_split_ids: {int(split_df['unique_id'].nunique())}")
    print(f"n_feature_ids: {int(feature_df['unique_id'].nunique())}")
    print(f"n_merged_ids: {int(merged['unique_id'].nunique())}")

    return merged


if __name__ == "__main__":
    print("*******************")
    print("Merging Datasets")
    args = parse_args()

    # exp_dir = os.path.join(BASE_DIR, "data", "training_sets", args.exp_name)
    # logger = ExperimentLogger(exp_dir)
    # logger.save_json("data-config", vars(args))

    if args.features == "ECIF": 
        feature_df = pd.read_csv(os.path.join(BASE_DIR, data/processed/raw_all_ecif_features.csv))
    elif args.features == "PLEC": 
        feature_df = pd.read_csv(os.path.join(BASE_DIR, data/processed/raw_all_plec_features.csv))
    
    #data_dir = os.path.join(exp_dir, "data")
    save_dir = os.path.join(BASE_DIR, "outputs", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for fold_idx in range(1, 6): # 5-fold splits
        
        #train_df = os.path.join(data_dir, f"{args.data_name}{fold_idx}_train.csv")
        train_df = os.path.join(BASE_DIR, "data/training_sets", f"{args.data_name}{fold_idx}_train.csv")
        #valid_df = os.path.join(data_dir, f"{args.data_name}{fold_idx}_valid.csv")
        valid_df = os.path.join(BASE_DIR, "data/training_sets", f"{args.data_name}{fold_idx}_valid.csv")

        train_outpath = os.path.join(save_dir, f"train_{fold_idx}_processed.csv")
        valid_outpath = os.path.join(save_dir, f"valid_{fold_idx}_processed.csv")
        
        train_merged = merge_datasets(train_df, feature_df, train_outpath, logger, "train", fold_idx)
        valid_merged = merge_datasets(valid_df, feature_df, valid_outpath, logger, "valid", fold_idx)