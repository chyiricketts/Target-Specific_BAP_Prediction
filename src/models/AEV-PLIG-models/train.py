import torch
import random
import time
import torch.nn as nn
from torch_geometric.loader import DataLoader
from helper.helpers import rmse, pearson, model_dict
from helper.utils import GraphDataset, init_weights
import os
import pandas as pd
import argparse
import numpy as np
import pickle
from pathlib import Path
import json
#from helper.logger import ExperimentLogger

class ExperimentLogger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.json_dir = os.path.join(exp_dir, "json")
        os.makedirs(self.json_dir, exist_ok=True)

    def _convert(self, obj):
        import numpy as np

        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)

        if isinstance(obj, dict):
            return {k: self._convert(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._convert(v) for v in obj]

        return obj

    def save_json(self, name, obj):
        path = os.path.join(self.json_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(self._convert(obj), f, indent=2)

BASE_DIR = Path.cwd().parents[1]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict(model, device, loader, y_scaler=None):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return y_scaler.inverse_transform(total_labels.numpy().flatten().reshape(-1,1)).flatten(), y_scaler.inverse_transform(total_preds.detach().numpy().flatten().reshape(-1,1)).flatten()


def train(model, device, train_loader, optimizer, epoch, loss_fn):
    log_interval = 100
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += (loss.item()*len(data.y))
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]'.format(epoch,
                                                             batch_idx * len(data.y),
                                                             len(train_loader.dataset),
                                                             100. * batch_idx / len(train_loader)))
    
    print("Loss for epoch {}: {:.4f}".format(epoch, total_loss/len(train_loader.dataset)))
    return total_loss/len(train_loader.dataset)



def _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, y_scaler, model_output_dir, model_file_name, history, patience):
    best_pc = -1.1
    patience = patience
    epochs_without_improvement = 0

    pcs = []
    for epoch in range(n_epochs):
    
        train_loss = train(model, device, train_loader, optimizer, epoch + 1, loss_fn)
        
        G, P = predict(model, device, valid_loader, y_scaler)
        current_pc = pearson(G, P)

        pcs.append(current_pc)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_pc"].append(current_pc)
        
        low = np.maximum(epoch-7,0)
        avg_pc = np.mean(pcs[low:epoch+1])

        """
        if(avg_pc > best_pc):
            torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))
            best_pc = avg_pc  
        """

        if avg_pc > best_pc:
            torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))
            best_pc = avg_pc
            epochs_without_improvement = 0
            print(f"Validation improved to {best_pc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        print('The current validation set Pearson correlation:', current_pc)

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
    G, P = predict(model, device, train_loader, y_scaler)
    final_train_pc = pearson(G, P)
    print("Final train Pearson:", final_train_pc)
    history["final_train_pc"] = final_train_pc

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--model', type=str, default='GATv2Net')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--head', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00012291937615434127)
    parser.add_argument('--activation_function', type=str, default='leaky_relu')
    parser.add_argument("--gnn_layers", type=int, default=5) # added myself
    parser.add_argument("--weight_decay", type=float, default=0) # added myself
    parser.add_argument("--patience", type=int, default=200)
    args = parser.parse_args()
    return args

def train_NN(args):
    start_time = time.time()

    # log args to json
    exp_dir = os.path.join(BASE_DIR, "scripts", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = ExperimentLogger(exp_dir)
    logger.save_json(f"{args.exp_name}_config", vars(args))

    modeling = model_dict[args.model]
    model_st = modeling.__name__
    
    batch_size = args.batch_size
    LR = args.lr
    n_epochs = args.epochs

    print('Train for {} epochs: '.format(n_epochs))

    exp_name = args.exp_name

    print(f"Running experiment {exp_name} on model {model_st}")
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(BASE_DIR, "scripts", args.exp_name)
    model_output_dir = os.path.join(exp_dir, "models")
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Running experiment {args.exp_name} on model {model_st}")
    
    train_data = GraphDataset(root=exp_dir, dataset="train", y_scaler=None)
    valid_data = GraphDataset(root=exp_dir, dataset="valid", y_scaler=train_data.y_scaler)
    #test_data  = GraphDataset(root=exp_dir, dataset="test",  y_scaler=train_data.y_scaler)

    #seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
    seeds = [100]
    for i,seed in enumerate(seeds):
        random.seed(seed)
        torch.manual_seed(int(seed))

        history = {
            "epoch": [],
            "train_loss": [],
            "val_pc": [],
            "final_train_pc": []
        }
                
        model_file_name = f"{timestr}_model_{model_st}_{exp_name}_seed{i}.model"

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4)
        #test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        if(torch.cuda.is_available()):
            print("GPU is available")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print('Device state:', device)

        model = modeling(node_feature_dim=train_data.num_node_features, edge_feature_dim=train_data.num_edge_features, config=args)
        model.apply(init_weights)

        # model stats to json
        model_stats = {
            "num_parameters": count_parameters(model),
            "node_features": train_data.num_node_features,
            "edge_features": train_data.num_edge_features
        }
        logger.save_json(f"{args.exp_name}_model_stats", model_stats) # instead of logger.save_json(f"model_stats_seed{i}", model_stats) -- would overwrite. but its redundant

        print("The number of node features is ", train_data.num_node_features)
        print("The number of edge features is ", train_data.num_edge_features)
        print("Trainable params:", count_parameters(model))
    
        weight_decay = args.weight_decay
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    
        model.to(device)
        _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, train_data.y_scaler, model_output_dir, model_file_name, history, args.patience)
        
        model.load_state_dict(torch.load(os.path.join(model_output_dir, model_file_name)))

        logger.save_json(f"{args.exp_name}_history_seed{i}", history)
        
        """
        G_test, P_test = predict(model, device, test_loader, train_data.y_scaler)

        if(i == 0):
            df_test = pd.DataFrame(data=G_test, index=range(len(G_test)), columns=['truth'])
        
        col = 'preds_' + str(i)
        df_test[col] = P_test
    
    df_test['preds'] = df_test.iloc[:,1:].mean(axis=1)
    """

    scaler_file = os.path.join(model_output_dir, f"{timestr}_model_{model_st}_{exp_name}.pickle")
    with open(scaler_file,'wb') as f:
        pickle.dump(train_data.y_scaler, f)
    
    #test_preds = np.array(df_test['preds'])
    #test_truth = np.array(df_test['truth'])
    #test_ens_pc = pearson(test_truth, test_preds)
    #test_ens_rmse = rmse(test_truth, test_preds)
    #print("Ensemble test PC:", test_ens_pc)
    #print("Ensemble test RMSE:", test_ens_rmse)

    # save to json
    #metrics = {
    #"test_pc": test_ens_pc,
    #"test_rmse": test_ens_rmse
    #}
    #logger.save_json(f"metrics_seed{i}", metrics)


    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    logger.save_json(f"{args.exp_name}_runtime", {
        "total_seconds": total_time,
        "hours": total_time / 3600
    })




if __name__ == "__main__":
    start_time = time.time()
    
    args = parse_args()
    
    train_NN(args)
    
    print("Total time is %s seconds" % (time.time() - start_time))

