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
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import Dropout
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv, Linear, global_mean_pool, BatchNorm


parser = argparse.ArgumentParser(description='itlubber submit args')
parser.add_argument('--seed', type=int, default=666, help='随机种子')
parser.add_argument('--epochs', type=int, default=4, help='训练多少个epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size 的大小')
parser.add_argument('--cuda', type=int, default=0, help='GPU的序号，默认第0块卡训练')
parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
parser.add_argument('--noise', type=float, default=0.05, help='数据噪声')
parser.add_argument('--feature_fraction', type=float, default=0.7, help='特征选择比例')
parser.add_argument('--clip', type=float, default=10.0, help='梯度裁剪')
parser.add_argument('--neg_weight', type=float, default=0.7, help='异常样本权重')
parser.add_argument('--edge_num', type=int, default=1, help='生成边的数量')
parser.add_argument('--attack', type=str, default="", help='异常样本权重')
parser.add_argument('--epsilon', type=float, default=1.0, help='异常样本权重')
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


def save_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)
        
        
def read_pickle(file, f, pseudo=0):
    """
    读取训练集和测试集数据
    """
    if f.endswith(".pkl"):
        data, metadata = load_pickle(f"data/{file}/{f}")
        data = pd.DataFrame(data, columns=['volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp','timestamp'])
        data["mileage"] = metadata.get("mileage")
        if metadata.get("label"):
            data["label"] = 1 if metadata.get("label") == "10" else 0
        else:
            data["label"] = pseudo
        
        data["file_name"] = f
        return data


def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)

    if n_total == 0 or offset < 1:
        return [], full_list

    if shuffle:
        random.shuffle(full_list)

    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    
    return sublist_1, sublist_2


# 模型参数设定
batch_size = args.batch_size
seed_everything(seed=args.seed)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")


# 读取训练集和测试集文件
# 如果需要修改数据文件地址，请修改下方读取文件位置
train = [read_pickle("Train", f) for f in tqdm(os.listdir("data/Train/"), desc="load train :::")]
test = [read_pickle("Test_A", f) for f in tqdm(os.listdir("data/Test_A/"), desc="load  test :::")]
# 测试集最终提交的测试集文件列表
train_index = [data["file_name"][0] for data in train]
test_index = [data["file_name"][0] for data in test]


# 转换原始数据为时序图结构
# 生成邻接矩阵
# source = []
# target = []
# for i in range(args.edge_num):
#     source.extend([j for j in range(256 - (i + 1))])
#     target.extend([j for j in range(i + 1, 256)])

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
    # if os.path.exists(f"cache/{mode}.pkl"):
    #     return load_pickle(f"cache/{mode}.pkl")
    
    graphs = []
    for row_data in tqdm(datasets, desc=f"process {mode} graph :::"):
        data = row_data.copy()
        data["timestamp"] = (data["timestamp"] - data["timestamp"].min()).astype(int)
        data["timestamp_diff"] = data["timestamp"].diff().fillna(0).apply(lambda x: math.log(x+1, 60))
        x = torch.FloatTensor(data.drop(columns=[col for col in ["file_name", "label"] if col in data.columns]).values.tolist())
        edge_index = torch.LongTensor([source, target]) # ([[i for i in range(255)], [i + 1 for i in range(255)]])
        edge_weight = data["timestamp"].loc[target].values - data["timestamp"].loc[source].values
        edge_attr = torch.FloatTensor(edge_weight)
        # edge_attr = torch.FloatTensor(data["timestamp"].diff().fillna(0).apply(lambda x: math.log(x+1, 60)).tolist()[1:])

        if mode != "test":
            y = torch.LongTensor([data["label"][0]])
            grpha = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            grpha = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        graphs.append(grpha)
        
    # save_pickle(graphs, f"cache/{mode}.pkl")
        
    return graphs


# 转换数据
graph_train = process_graph(train, mode="train")
graph_test = process_graph(test, mode="test")


# 定义图神经网络 dataset 和 dataloader
class GraphaDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data, self.slices = self.collate(data_list)
        

dateset_train = GraphaDataset(graph_train)
dateset_test = GraphaDataset(graph_test)

loader_train = DataLoader(dateset_train, batch_size=batch_size, shuffle=True)
# loader_vail = DataLoader(dateset_train, batch_size=batch_size, shuffle=False)
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
    

class FGM:
    def __init__(self, model, emb_name='conv1.lin.weight'):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}

    def attack(self, epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 模型初始化
num_epoches = args.epochs
writer = SummaryWriter()
model = GCNNet(dateset_train, 4, 256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.025)
t_total = (len(dateset_train) // batch_size + 1) * num_epoches
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, final_div_factor=100, div_factor=10, max_lr=args.lr, total_steps=t_total, pct_start=0.3)
criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0 - args.neg_weight, args.neg_weight]).to(device))
fgm = FGM(model, emb_name="conv")


# 定义网络训练、验证、推理函数
def train(loader, epoch=0):
    model.train()
    data_iter = tqdm(enumerate(loader), desc="pyg model training :::", total=len(loader))
    
    for n_iter, data in data_iter:
        data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, data.y)
        
        loss.backward()
        
        writer.add_scalar('train_loss', loss.cpu().item(), len(loader) * epoch + n_iter)
        
        if args.attack == "FGM":
            fgm.attack(epsilon=args.epsilon)
            loss_adv = criterion(model(data), data.y)
            loss_adv.backward()
            fgm.restore()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step()
        scheduler.step()
        
        data_iter.set_postfix(loss='{:.4f}'.format(loss.cpu().item()))
        
        
def test(loader, mode="train", epoch=0):
    model.eval()
    preps = []
    trues = []
    with torch.no_grad():
        for n_iter, data in tqdm(enumerate(loader), desc=f"pyg {mode} evaluate :::", total=len(loader)):
            trues.extend(data.y.cpu().numpy().tolist())
            data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            # pred = F.softmax(out)[:, 1]
            pred = out[:, 1]
            preps.extend(pred.cpu().numpy().tolist())
            
            writer.add_scalar(f'{mode}_auc', roc_auc_score(data.y.cpu().numpy().tolist(), pred.cpu().numpy().tolist()), len(loader) * epoch + n_iter)

    return roc_auc_score(trues, preps)


def inference(loader):
    model.eval()

    results = []
    with torch.no_grad():
        for data in tqdm(loader, desc="pyg model inference :::"):
            data.to(device)
            out = model(data)
            # pred = F.softmax(out)[:, 1]
            pred = out[:, 1]
            results.extend(pred.cpu().numpy().tolist())

    return results


def inference_output(loader):
    model.eval()

    results = []
    with torch.no_grad():
        for data in tqdm(loader, desc="pyg model embedding :::"):
            data.to(device)
            out = model(data)
            results.extend(out.cpu().numpy().tolist())

    return results


# 模型训练
for epoch in range(num_epoches):
    train(loader_train)
    train_auc = test(loader_train, epoch=epoch, mode="train")
    
    print(f'epoch: {epoch:03d}, train auc: {train_auc:.4f}')


# 模型推理
pred = inference(loader_test)

# 模型保存
torch.save(model, "vloong_anomaly_detection_model.torch")

# 生成特征
# output_vail = inference_output(loader_vail)
# output_test = inference_output(loader_test)

# save_pickle(dict(zip(train_index + test_index, output_vail + output_test)), "cache/output_features.pkl")

# 生成结果文件
pd.DataFrame(list(zip(test_index, pred)), columns=["file_name", "score"]).to_csv("result.csv", index=False)


# tensorboard --logdir=f"runs" --port=8014 --bind_all
