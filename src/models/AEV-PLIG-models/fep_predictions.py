import pandas as pd
import pickle
import torch
import numpy as np
import os
import argparse
import time
import warnings
from torch_geometric.loader import DataLoader
from helper.utils import GraphDatasetPredict
from helper.helpers import model_dict
from helper.logger import ExperimentLogger
import json
from pathlib import Path

BASE_DIR = Path.cwd().parents[1]
device = "cpu"

# Suppress Torchani warnings
warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.ase will not be available")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.data will not be available")


def predict(model, device, loader, y_scaler=None):
    model.eval()
    model.to(device)
    total_preds = torch.Tensor().to(device)
    total_graph_ids = torch.IntTensor().to(device)
    
    print('Make prediction for {} samples...'.format(len(loader.dataset)))

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output), 0)
            total_graph_ids = torch.cat((total_graph_ids, data.y.view(-1, 1)), 0)

    return total_graph_ids.cpu().numpy().flatten(), y_scaler.inverse_transform(total_preds.cpu().detach().numpy().flatten().reshape(-1,1)).flatten()



def make_predictions(config):
    print("Make predictions\n")
    runtime_t1 = time.time()
    
    exp_dir = os.path.join(BASE_DIR, "scripts", config.exp_name)
    model_dir = os.path.join(exp_dir, "models")

    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".model")])
    model_base_name = model_files[0].rsplit("_seed", 1)[0]
    print("Predictions from model ensemble: ", model_base_name)

    scaler_path = os.path.join(model_dir, model_base_name + ".pickle")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    """
    Create .pt file from graphs
    """
    data = pd.read_csv(os.path.join(BASE_DIR, "data/processed", "fep_prediction_info.csv"))

    with open(os.path.join(BASE_DIR, "data/processed", "fep_graphs.pickle"), "rb") as handle:
        graphs_dict = pickle.load(handle)

    data["graph_id"] = range(len(data))
    test_ids = list(data["unique_id"])
    test_graph_ids = list(data["graph_id"])

    # internal .pt file -- only makes if existing one is not there
    test_data = GraphDatasetPredict(
        root=os.path.join(BASE_DIR, "fep", "data"),
        dataset='fep', 
        ids=test_ids, 
        graph_ids=test_graph_ids, 
        graphs_dict=graphs_dict)

    """
    Make predictions
    """
    test_loader = DataLoader(test_data, batch_size=len(data), shuffle=False)

    modeling = model_dict['GATv2Net']
    model = modeling(node_feature_dim=test_data.num_node_features, edge_feature_dim=test_data.num_edge_features, config=config)

    for i, mf in enumerate(model_files):
        path = os.path.join(model_dir, mf)
        model.load_state_dict(torch.load(path, map_location="cpu"))

        graph_ids_test, P_test = predict(model, device, test_loader, scaler)

        if(i == 0):
            df_test = pd.DataFrame(data=graph_ids_test, index=range(len(graph_ids_test)), columns=['graph_id'])

        col = 'preds_' + str(i)
        df_test[col] = P_test

    df_test['preds'] = df_test.iloc[:,1:].mean(axis=1)

    data = data.merge(df_test, on='graph_id', how='left')

    """
    Save predictions
    Globally and per-experiment
    """
    print("Saving predictions\n")
    data.to_csv(os.path.join(exp_dir, config.exp_name + "_predictions.csv"), index=False)
    data.to_csv(os.path.join(BASE_DIR, "predictions", config.exp_name + "_predictions.csv"), index=False)


    logger = ExperimentLogger(exp_dir)
    logger.save_json("config", vars(config))

    total_time = time.time() - runtime_t1

    logger.save_json("prediction_summary", {
    "experiment": {
        "exp_name": config.exp_name,
        "gnn_layers": config.gnn_layers,
        "hidden_dim": config.hidden_dim,
        "head": config.head,
        "activation_function": config.activation_function,
        "model_prefix": model_base_name,
        "test_data": "fep",
        "test_data_complexes": len(data)
    },
    "runtime": {
        "total_seconds": total_time,
        "hours": total_time / 3600,
        "seconds_per_sample": total_time / len(data)
    }
    })


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument('--gnn_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--head', type=int, default=3)
    parser.add_argument('--activation_function', type=str, default='leaky_relu')

    return parser.parse_args()
        
if __name__ == "__main__":    
    config = parse_args()

    t1 = time.time()
    make_predictions(config)
    print("Time to make predictions:", time.time()-t1)

