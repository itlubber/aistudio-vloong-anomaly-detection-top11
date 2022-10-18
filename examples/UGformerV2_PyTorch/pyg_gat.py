import os
import gc
import re
import math
import time
import toad
import json
import pickle
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import seaborn as sns
import networkx as nx
from datetime import datetime
from functools import lru_cache
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_geometric.nn import Linear, HeteroConv, GraphConv, GAT, RGCNConv, BatchNorm, GCN
from torch_geometric.nn.models import JumpingKnowledge

# from util import 
from UGformerV2 import FullyConnectedGT_UGformerV2, label_smoothing


warnings.filterwarnings("ignore")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_graphs(dataset, edge_lag=1, weight_base=1.05):
    if os.path.exists(f"../cache/{os.path.basename(dataset)}.pkl"):
        return load_pickle(f"../cache/{os.path.basename(dataset)}.pkl")

    pickle_files = glob(dataset + "/*.pkl")

    graphs = []
    label_dict = {"00": 0, "10": 1}

    source = []
    target = []
    for i in range(1, edge_lag + 1):
        source.extend([i for i in range(256 - i)])
        target.extend([i for i in range(i, 256)])

    for file in tqdm(pickle_files, desc=f"load {dataset} data :::"):
        features, metadata = load_pickle(file)
        
        mileage = metadata.get("mileage")
        label = label_dict[metadata.get("label", "00")]

        features = pd.DataFrame(features, columns=['volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp','timestamp'])
        features["mileage"] = mileage
        features["file_name"] = os.path.basename(file)
        features["timestamp"] = (features["timestamp"] - features["timestamp"].min()).astype(int)
        features["timestamp_diff"] = features["timestamp"].diff().fillna(0) # .apply(lambda x: math.log(x+1, 60))

        x = torch.FloatTensor(features.drop(columns=[col for col in ["file_name", "label"] if col in features.columns]).values.tolist())
        edge_index = torch.LongTensor([source, target])
        edge_weight = np.power(weight_base, (features["timestamp"].loc[source].values - features["timestamp"].loc[target].values + 1))
        edge_attr = torch.FloatTensor(edge_weight)

        grpha = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)

        graphs.append(grpha)

    save_pickle(graphs, f"../cache/{os.path.basename(dataset)}.pkl")

    return graphs


def separate_data(graphs, fold_idx, seed=3407):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."

    labels = [graph.y for graph in graphs]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    
    train_idx, test_idx = idx_list[fold_idx]
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    return train_graphs, test_graphs


def get_Adj_matrix(graph):
    Adj_block_idx = torch.LongTensor(graph.edge_index)
    Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

    num_node = len(graph.x)
    self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
    elem = torch.ones(num_node)
    Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
    Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

    Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([num_node, num_node]))

    return Adj_block


def get_data(graph):
    node_features = graph.x
    Adj_block = get_Adj_matrix(graph)
    return Adj_block.to(device), node_features.to(device), torch.LongTensor([graph.y]).to(device)


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


# Parameters
# //////////////////////////////////////////////////////////////////////////////////
parser = argparse.ArgumentParser("UGformer", formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--train", default="../data/Train", help="")
parser.add_argument("--test", default="../data/Test_A", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='PTC', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--nhead", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=1, type=int, help="Number of self-attention layers within each UGformer layer")
parser.add_argument("--ff_hidden_size", default=256, type=int, help="The hidden size for the feedforward layer")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
parser.add_argument("--seed", type=int, default=3407, help="The random seed.")
args = parser.parse_args()

print(args)


# seed everthing
# //////////////////////////////////////////////////////////////////////////////////
seed_everything(seed=args.seed)


# Load data
# //////////////////////////////////////////////////////////////////////////////////
print("Loading data...")

num_classes = 2
train_graphs = load_graphs(args.train)
test_graphs = load_graphs(args.test)

if args.fold_idx >= 0:
    trin_graphs, vail_graphs = separate_data(train_graphs, args.fold_idx, seed=args.seed)

feature_dim_size = train_graphs[0].x.shape[1]

print("Loading data... finished!")


# Credit Model
# //////////////////////////////////////////////////////////////////////////////////
model = FullyConnectedGT_UGformerV2(
    feature_dim_size=feature_dim_size, 
    ff_hidden_size=args.ff_hidden_size,
    num_classes=num_classes, 
    dropout=args.dropout,
    num_self_att_layers=args.num_timesteps,
    num_GNN_layers=args.num_hidden_layers,
    nhead=args.nhead
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def train(graphs):
    model.train() # Turn on the train mode
    total_loss = 0.
    idxs = np.arange(len(graphs))
    np.random.shuffle(idxs)
    for idx in idxs:
        Adj_block, node_features, graph_label = get_data(graphs[idx]) # one graph per step. should modify to use "padding" (for node_features and Adj_block) within a batch size???
        # import pdb; pdb.set_trace()
        graph_label = label_smoothing(graph_label, num_classes)
        optimizer.zero_grad()
        prediction_score = model.forward(Adj_block, node_features)
        loss = criterion(prediction_score, graph_label)
        # loss = cross_entropy(torch.unsqueeze(prediction_score, 0), graph_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent the exploding gradient problem
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate(graphs):
    model.eval() # Turn on the evaluation mode

    with torch.no_grad():
        # evaluating
        prediction_output = []
        for i in range(0, len(graphs)):
            Adj_block, node_features, graph_label = get_data(graphs[i])
            prediction_score = model.forward(Adj_block, node_features).detach()
            prediction_output.append(torch.unsqueeze(prediction_score, 0))

    prediction_output = torch.cat(prediction_output, 0)
    predictions = prediction_output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in graphs]).to(device)
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    acc_test = correct / float(len(graphs))

    return acc_test


cost_loss = []
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss = train(trin_graphs)
    cost_loss.append(train_loss)
    acc_test = evaluate(vail_graphs)
    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | test acc {:5.2f} | '.format(epoch, (time.time() - epoch_start_time), train_loss, acc_test*100))
