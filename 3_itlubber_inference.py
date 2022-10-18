# !pip install tqdm torch tensorboardX torch_geometric
# 导入相关的库
import warnings

warnings.filterwarnings("ignore")

import os
import gc
import math
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv, Linear, global_mean_pool, BatchNorm


parser = argparse.ArgumentParser(description='itlubber submit args')
parser.add_argument('--test_dir', type=str, default="data/Test_A/", help='训练集数据的文件夹相对路径或绝对路径')
parser.add_argument('--model', type=str, default="vloong_anomaly_detection_model_pseudo.torch", help='训练保存好的模型文件路径')
parser.add_argument('--seed', type=int, default="666", help='随机种子')
parser.add_argument('--batch_size', type=int, default=64, help='推理的 batch size 大小')
parser.add_argument('--cuda', type=int, default=0, help='GPU的序号，默认第0块卡训练')
parser.add_argument('--edge_num', type=int, default=1, help='生成边的数量')

parser.add_argument('--noise', type=float, default=0.05, help='数据噪声')
parser.add_argument('--feature_fraction', type=float, default=0.7, help='特征选择比例')
args = parser.parse_args()


print(args)


def seed_everything(seed: int):
    """
    固定随机种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_pickle(file):
    """
    加载 pickle 文件
    """
    with open(file, "rb") as f:
        return pickle.load(f)


def read_pickle(dirname, f, pseudo=0):
    """
    读取训练集和测试集数据
    """
    if f.endswith(".pkl"):
        data, metadata = load_pickle(os.path.join(dirname, f))
        data = pd.DataFrame(data, columns=['volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp','timestamp'])
        data["mileage"] = metadata.get("mileage")
        if metadata.get("label"):
            data["label"] = 1 if metadata.get("label") == "10" else 0
        else:
            data["label"] = pseudo
        
        data["file_name"] = f
        return data


# 模型参数设定
batch_size = args.batch_size
seed_everything(seed=args.seed)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")


# 读取训练集和测试集文件
# 如果需要修改数据文件地址，请修改下方读取文件位置
test = [read_pickle(args.test_dir, f) for f in tqdm(os.listdir(args.test_dir), desc="load  test :::")]
# 测试集最终提交的测试集文件列表
test_index = [data["file_name"][0] for data in test]


# 转换原始数据为时序图结构
# 生成邻接矩阵
source = []
target = []
for i in range(args.edge_num):
    if i == 0:
        source.extend([j for j in range(256 - (i + 1))])
        target.extend([j for j in range(i + 1, 256)])
    else:
        p = random.random() < (1.0 / args.edge_num * (i + 1))
        source.extend([j for j in range(256 - (i + 1)) if p])
        target.extend([j for j in range(i + 1, 256) if p])
    

def process_graph(datasets, mode="train"):
    """
    时序数据转换图结构数据
    """
    graphs = []
    for row_data in tqdm(datasets, desc=f"process {mode} graph :::"):
        data = row_data.copy()
        data["timestamp"] = (data["timestamp"] - data["timestamp"].min()).astype(int)
        data["timestamp_diff"] = data["timestamp"].diff().fillna(0).apply(lambda x: math.log(x+1, 60))
        x = torch.FloatTensor(data.drop(columns=[col for col in ["file_name", "label"] if col in data.columns]).values.tolist())
        edge_index = torch.LongTensor([source, target])
        edge_weight = data["timestamp"].loc[target].values - data["timestamp"].loc[source].values
        edge_attr = torch.FloatTensor(edge_weight)

        if mode != "test":
            y = torch.LongTensor([data["label"][0]])
            grpha = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            grpha = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        graphs.append(grpha)
        
    return graphs


# 转换数据
graph_test = process_graph(test, mode="test")


# 定义图神经网络 dataset 和 dataloader
class GraphaDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data, self.slices = self.collate(data_list)
        

dateset_test = GraphaDataset(graph_test)
loader_test = DataLoader(dateset_test, batch_size=batch_size, shuffle=False)


# 定义图神经网络模型
class GCNNet(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden)
        self.bn1 = BatchNorm(hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.bn1.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.training:
            x = self.add_noise(x)
        x = F.dropout(x, p=1.0-args.feature_fraction, training=self.training)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    @staticmethod
    def add_noise(x, perturb_noise=args.noise):
        perturb = torch.empty_like(x).uniform_(-perturb_noise, perturb_noise)
        return x + perturb

    def __repr__(self):
        return self.__class__.__name__


model = torch.load(args.model)


def inference(loader):
    model.eval()

    results = []
    with torch.no_grad():
        for data in tqdm(loader, desc="pyg model inference :::"):
            data.to(device)
            out = model(data)
            pred = out[:, 1]
            results.extend(pred.cpu().numpy().tolist())

    return results


# 模型推理
pred = inference(loader_test)


# 生成结果文件
pd.DataFrame(list(zip(test_index, pred)), columns=["file_name", "score"]).to_csv("inference_result.csv", index=False)
