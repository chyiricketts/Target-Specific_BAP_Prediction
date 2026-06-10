# for eval all these models on the FEP features. 
# for models

import pandas as pd
import argparse
import os
import numpy as np
from helper.load_model import load_model

BASE_DIR = "/rds/general/user/cr725/home/aev-plig_research"

def evaluate(models, df, feature_type):

    if feature_type == "ECIF": 
        feature_cols = [c for c in df.columns if c.startswith("ECIF_")]
    elif feature_type == "PLEC": 
        feature_cols = [c for c in df.columns if c.startswith("PLEC_")]
    else: 
        raise ValueError(f"Invalid feature type: {feature_type}. Use 'ECIF' or 'PLEC'.")

    X = df[feature_cols].values
    y = df["dg_exp"].values

    all_preds = np.column_stack([model.predict(X) for model in models])
    preds = all_preds.mean(axis=1)

    results_df = pd.DataFrame({
        "unique_id": df["unique_id"].values,
        "dg_exp": y,
        "preds": preds
    })

    for i in range(all_preds.shape[1]):
        results_df[f"preds_{i+1}"] = all_preds[:, i]

    return results_df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--model", choices=["xgb", "rf", "svr"], required=True)
    parser.add_argument("--feature_type", type=str, default="ECIF") 

    return parser.parse_args()


if __name__ == "__main__":
    print("\n******************")
    print("Evaluating model")
    args = parse_args()

    BASE_EXP = os.path.join(BASE_DIR,"ts_other-models_CV","experiments", args.exp_name)

    models = []


    # for folds
    for fold_idx in range(1, 6):
        
        filename = (f"{args.model}_fold_{fold_idx}.json" if args.model == "xgb"
                    else f"{args.model}_fold_{fold_idx}.joblib")
        
        model_path = os.path.join(BASE_EXP, "models_cv", filename)

        model = load_model(args.model, model_path)
        models.append(model)

    if args.feature_type == "ECIF": 
        fep_df = pd.read_csv(os.path.join(BASE_DIR, "ts_other-models_CV", "ECIF_data", "fep_ecif_features.csv"))
    elif args.feature_type == "PLEC": 
        fep_df = pd.read_csv(os.path.join(BASE_DIR, "ts_other-models_CV", "PLEC_data", "fep_plec_features.csv"))
    else: 
        raise ValueError(f"Invalid feature type: {args.feature_type}. Use 'ECIF' or 'PLEC'.")

    results_df = evaluate(models, fep_df, args.feature_type)

    results_df.to_csv(os.path.join(BASE_DIR, "ts_other-models_CV", "predictions", args.exp_name + "_cv_predictions.csv"), index=False)
    results_df.to_csv(os.path.join(BASE_EXP, args.exp_name + "_cv_predictions.csv"), index=False)




    models = []


    # for seeds
    seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
    for seed in seeds:
        
        filename = (f"{args.model}_seed_{seed}.json" if args.model == "xgb"
                else f"{args.model}_seed_{seed}.joblib")
        model_path = os.path.join(BASE_EXP, "models_seeded", filename)
        model = load_model(args.model, model_path)
        models.append(model)

    if args.feature_type == "ECIF": 
        fep_df = pd.read_csv(os.path.join(BASE_DIR, "ts_other-models_CV", "ECIF_data", "fep_ecif_features.csv"))
    elif args.feature_type == "PLEC": 
        fep_df = pd.read_csv(os.path.join(BASE_DIR, "ts_other-models_CV", "PLEC_data", "fep_plec_features.csv"))
    else: 
        raise ValueError(f"Invalid feature type: {args.feature_type}. Use 'ECIF' or 'PLEC'.")

    results_df = evaluate(models, fep_df, args.feature_type)

    results_df.to_csv(os.path.join(BASE_DIR, "ts_other-models_CV", "predictions", args.exp_name + "_seeded_predictions.csv"), index=False)
    results_df.to_csv(os.path.join(BASE_EXP, args.exp_name + "_seeded_predictions.csv"), index=False)

