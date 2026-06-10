# for models

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib
import pandas as pd
import argparse
import json
import os
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from helper.logger import ExperimentLogger
from sklearn.model_selection import cross_val_score
import optuna
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
import matplotlib.pyplot as plt
import random

BASE_DIR = "/rds/general/user/cr725/home/aev-plig_research"

def get_model(name, params):
    if name == "xgb":
        return XGBRegressor(
            **params,
            n_jobs=-1,
            objective="reg:squarederror"
        )

    elif name == "rf":
        return RandomForestRegressor(
            n_jobs=-1,
            **params
        )

    elif name == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(**params))
        ])
    
    else: raise ValueError(f"Unknown model: {name}")


# helper method
def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    pearson = pearsonr(y, preds)[0]

    print(
        f"[{split_name}] "
        f"R2={r2:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"Pearson={pearson:.4f}"
    )

    return {
        "r2": r2,
        "rmse": rmse,
        "pearson": pearson
    }

# getting the train and valid data per fold
def load_fold(exp_name, fold_idx):

    processed_dir = os.path.join(BASE_DIR, "ts_other-models_CV", "experiments", exp_name, "processed")
    train_df = pd.read_csv(os.path.join(processed_dir,  f"train_{fold_idx}_processed.csv"))
    valid_df = pd.read_csv(os.path.join(processed_dir,  f"valid_{fold_idx}_processed.csv"))

    return train_df, valid_df


def train_single_fold(model_name, params, train_df, valid_df, feature_type): 
    if feature_type == "ECIF": 
        feature_cols = [c for c in train_df.columns if c.startswith("ECIF_")]
    elif feature_type == "PLEC": 
        feature_cols = [c for c in train_df.columns if c.startswith("PLEC_")]
    else: 
        raise ValueError(f"Invalid feature type: {feature_type}. Use 'ECIF' or 'PLEC'.")

    X_train = train_df[feature_cols].values
    y_train = train_df["pK"].values

    X_valid = valid_df[feature_cols].values
    y_valid = valid_df["pK"].values

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Valid: X={X_valid.shape}, y={y_valid.shape}")

    if len(X_train) == 0:
        print("No training data")
        return

    model = get_model(model_name, params)

    model.fit(X_train, y_train)

    train_metrics = evaluate(model, X_train, y_train, "train")
    valid_metrics = evaluate(model, X_valid, y_valid, "valid")

    return {"train": train_metrics, "valid": valid_metrics}, model


def cross_validate(exp_name, model_name, params, logger, feature_type):

    fold_metrics = []

    for fold_idx in range(1,6):
        print(f"\n===== Fold {fold_idx} =====")
        train_df, valid_df = load_fold(exp_name, fold_idx)
        metrics, model = train_single_fold(model_name, params, train_df, valid_df, feature_type)
        fold_metrics.append(metrics)

        model_outpath = os.path.join(BASE_DIR, "ts_other-models_CV", "experiments", exp_name, "models_cv")
        os.makedirs(model_outpath, exist_ok=True)

        if model_name == "xgb":
            model.save_model(os.path.join(model_outpath, f"{model_name}_fold_{fold_idx}.json"))
        else:
            joblib.dump(model, os.path.join(model_outpath, f"{model_name}_fold_{fold_idx}.joblib"))

    summary = aggregate_metrics(fold_metrics)

    logger.save_json(f"{model_name}_cv_summary", {
        "summary": summary,
        "folds": fold_metrics,
        "params": params
    })

    return {
        "summary": summary,
        "folds": fold_metrics}


def aggregate_metrics(fold_metrics):

    valid_rmse =     [m["valid"]["rmse"] for m in fold_metrics]
    valid_pearson =  [m["valid"]["pearson"] for m in fold_metrics]
    valid_r2 =       [m["valid"]["r2"] for m in fold_metrics]

    return {
        "mean_rmse": np.mean(valid_rmse),
        "std_rmse": np.std(valid_rmse),

        "mean_pearson": np.mean(valid_pearson),
        "std_pearson": np.std(valid_pearson),

        "mean_r2": np.mean(valid_r2),
        "std_r2": np.std(valid_r2),
    }


# for optuna on xgb
def xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

# optuna on rf
def rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": True
    }

# optuna for svr
def svr_params(trial):
    return {
        "C": trial.suggest_float("C", 0.1, 100, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": "rbf"
    }

# main optuna function
def objective(trial, model_name, exp_name, feature_type):
    if model_name == "xgb":
        params = xgb_params(trial)
    elif model_name == "rf":
        params = rf_params(trial)
    elif model_name == "svr":
        params = svr_params(trial)

    return cv_score(model_name, params, exp_name, feature_type)


def cv_score(model_name, params, exp_name, feature_type):
    fold_scores = []

    for fold in range(1, 6):
        train_df, valid_df = load_fold(exp_name, fold)
        
        if feature_type == "ECIF": 
            feature_cols = [c for c in train_df.columns if c.startswith("ECIF_")]
        elif feature_type == "PLEC": 
            feature_cols = [c for c in train_df.columns if c.startswith("PLEC_")]
        else: 
            raise ValueError(f"Invalid feature type: {feature_type}. Use 'ECIF' or 'PLEC'.")
        
        X_train = train_df[feature_cols].values
        y_train = train_df["pK"].values

        X_valid = valid_df[feature_cols].values
        y_valid = valid_df["pK"].values

        model = get_model(model_name, params)
        model.fit(X_train, y_train)

        preds = model.predict(X_valid)
        fold_scores.append(r2_score(y_valid, preds))

    return np.mean(fold_scores)

# combining train and valid for final model
def load_full_train(exp_dir):
    train_df = pd.read_csv(os.path.join(exp_dir, "processed", "train_1_processed.csv"))
    valid_df = pd.read_csv(os.path.join(exp_dir, "processed", "valid_1_processed.csv"))

    return pd.concat([train_df, valid_df], ignore_index=True)


def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--model", choices=["xgb", "rf", "svr"], required=True)
    parser.add_argument("--params", type=str, default="{}")  # JSON string
    parser.add_argument("--feature_type", type=str, default="ECIF") 

    return parser.parse_args()


if __name__ == "__main__":
    print("\n******************")
    print("Training Model")
    args = parse_args()

    exp_dir = os.path.join(BASE_DIR, "ts_other-models_CV", "experiments", args.exp_name)
    logger = ExperimentLogger(exp_dir)
    os.makedirs(os.path.join(exp_dir, "models_cv"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models_seeded"), exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, args.model, args.exp_name, args.feature_type),
        n_trials=50
    )
    logger.save_json("train_best_params", study.best_params)

    # visualization
    os.makedirs(os.path.join(exp_dir, "figures"), exist_ok=True)
    fig = plot_optimization_history(study); fig.figure.savefig(os.path.join(exp_dir, "figures", "optimization_history.png"), bbox_inches="tight")
    fig = plot_param_importances(study); fig.figure.savefig(os.path.join(exp_dir, "figures", "param_importance.png"), bbox_inches="tight")
    fig = plot_parallel_coordinate(study); fig.figure.savefig(os.path.join(exp_dir, "figures", "parallel_coords.png"), bbox_inches="tight")
    #fig = plot_parallel_coordinate(study, params=["learning_rate", "max_depth", "n_estimators", "subsample"]); fig.figure.savefig(os.path.join(exp_dir, "figure", "parallel_coords2.png"), bbox_inches="tight")
    #fig = plot_contour(study, params=["learning_rate", "n_estimators"]); fig.figure.savefig(os.path.join(exp_dir, "figure", "contour_lr_estimators.png"), bbox_inches="tight")
    #fig = plot_contour(study, params=["max_depth", "min_child_weight"]); fig.figure.savefig(os.path.join(exp_dir, "figure", "contour_depth_childweight.png"), bbox_inches="tight")
    #plot_slice(study, params=["learning_rate", "max_depth", "n_estimators"]); plt.gcf().savefig(os.path.join(exp_dir, "json", "plot_slice.png"), bbox_inches="tight")
    #plt.close()
    
    base_params = study.best_params.copy()
    
    results = cross_validate(
        exp_name=args.exp_name, model_name=args.model, params=base_params, 
        logger=logger, feature_type=args.feature_type )

    # 10 seeds on all training data
    # later change this to argment specific
    train_df = load_full_train(exp_dir)

    if args.feature_type == "ECIF":
        feature_cols = [c for c in train_df.columns if c.startswith("ECIF_")]
    elif args.feature_type == "PLEC":
        feature_cols = [c for c in train_df.columns if c.startswith("PLEC_")]
    else:
        raise ValueError()

    X_train = train_df[feature_cols].values
    Y_train = train_df["pK"].values

    seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
    for seed in seeds:
        print(f"\n===== Training seed {seed} =====")

        np.random.seed(seed)
        random.seed(seed)

        params = base_params.copy()

        if args.model != "svr": 
            params["random_state"] = seed
            
        model = get_model(args.model, params)

        model.fit(X_train, Y_train)

        filename = (
            f"{args.model}_seed_{seed}.json"
            if args.model == "xgb"
            else f"{args.model}_seed_{seed}.joblib"
        )
        path = os.path.join(exp_dir, "models_seeded", filename)

        if args.model == "xgb":
            model.save_model(path)
        else:
            joblib.dump(model, path)