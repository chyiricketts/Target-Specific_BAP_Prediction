import pandas as pd
import pickle
import argparse
import json
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path
#from helper.logger import ExperimentLogger
#from helper.utils import GraphDataset


# originally from helper.utils and logger
def init_weights(layer):
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is True:
            torch.nn.init.zeros_(layer.bias)

class GraphDataset(InMemoryDataset):
    def __init__(self, root='data', dataset=None,
                 ids=None, y=None, graphs_dict=None, y_scaler=None):

        super(GraphDataset, self).__init__(root)
        self.dataset = dataset
        #torch.serialization.add_safe_globals([GraphDataset])
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            #self.data, self.slices = torch.load(self.processed_paths[0])
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])

        else:
            self.process(ids, y, graphs_dict)
            #self.data, self.slices = torch.load(self.processed_paths[0])
            self.load(self.processed_paths[0])
        
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(self._data.y, (self.__len__(),1)))
        self.y_scaler = y_scaler
        self._data.y = [torch.tensor(element[0]).float() for element in self.y_scaler.transform(np.reshape(self._data.y, (self.__len__(),1)))]
        

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, y, graphs_dict):
        assert (len(ids) == len(y)), 'Number of datapoints and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            pdbcode = ids[i]
            label = y[i]
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            data_point = Data(x=torch.Tensor(np.array(features)),
                                   edge_index=torch.LongTensor(np.array(edge_index)).T,
                                   edge_attr=torch.Tensor(np.array(edge_features)),
                                   y=torch.FloatTensor(np.array([label])))
            
            data_list.append(data_point)

        print('Graph construction done. Saving to file.')
        #self.data, self.slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])
        #torch.save((self.data, self.slices), self.processed_paths[0])
        


class GraphDatasetPredict(InMemoryDataset):
    def __init__(self, root='data', dataset=None,
                 ids=None, graph_ids=None, graphs_dict=None):

        super(GraphDatasetPredict, self).__init__(root)
        self.dataset = dataset
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])

        else:
            self.process(ids, graph_ids, graphs_dict)
            self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, graph_ids, graphs_dict):
        assert (len(ids) == len(graph_ids)), 'Number of datapoints and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            pdbcode = ids[i]
            graph_id = graph_ids[i]
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            data_point = Data(x=torch.Tensor(np.array(features)),
                                   edge_index=torch.LongTensor(np.array(edge_index)).T,
                                   edge_attr=torch.Tensor(np.array(edge_features)),
                                   y=torch.IntTensor(np.array([graph_id])))
            
            data_list.append(data_point)

        print('Graph construction done. Saving to file.')
        self.save(data_list, self.processed_paths[0])


class ExperimentLogger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.json_dir = os.path.join(exp_dir, "json")
        os.makedirs(self.json_dir, exist_ok=True)

    def save_json(self, name, obj):
        path = os.path.join(self.json_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

BASE_DIR = Path.cwd().parents[1]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, required=True)

    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--valid_data", type=str, default=None)

    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    exp_dir = os.path.join(BASE_DIR, "scripts", args.exp_name)
    logger = ExperimentLogger(exp_dir)

    # load graphs
    graphs_path = os.path.join(exp_dir, "graphs.pkl")

    with open(graphs_path, "rb") as f:
        graphs_dict = pickle.load(f)

    print(f"Loaded {len(graphs_dict)} graphs")


    dfs = []

    # read split
    train_df = pd.read_csv(args.train_data)
    valid_df = pd.read_csv(args.valid_data)

    train_ids = train_df["unique_id"].tolist()
    train_y   = train_df["pK"].tolist()

    valid_ids = valid_df["unique_id"].tolist()
    valid_y   = valid_df["pK"].tolist()

    print("Split sizes:")
    print("Train:", len(train_ids))
    print("Valid:", len(valid_ids))

    # prepare the graphs within experimental folder
    print('Preparing train.pt...')
    train_data = GraphDataset(
        root=exp_dir,
        dataset="train",
        ids=train_ids,
        y=train_y,
        graphs_dict=graphs_dict
    )

    print('Preparing valid.pt...')
    valid_data = GraphDataset(
        root=exp_dir,
        dataset="valid",
        ids=valid_ids,
        y=valid_y,
        graphs_dict=graphs_dict,
        y_scaler=train_data.y_scaler
    )

    config = {
    "datasets_used": {
        "train": args.train_data is not None,
        "valid": args.valid_data is not None
    },
    "n_train": len(train_ids),
    "n_valid": len(valid_ids)
    }

    logger.save_json(f"{args.exp_name}_dataset_config", config)