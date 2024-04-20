import math
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from utils import extract_enclosing_subgraphs  
from models import DGCNN
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--k_ratio', type=float, default=0.6)
    parser.add_argument('--z_max', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='brightkite')
    

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == "brightkite":
        edge_index_tensor = torch.tensor(np.loadtxt('datasets/Brightkite_edges.txt', dtype=np.int64).T, dtype=torch.long)
    elif args.dataset == "adv":
        edge_index_tensor = torch.tensor(np.loadtxt('datasets/ADV_full.txt', dtype=np.int64).T, dtype=torch.long) - 1
    elif args.dataset == "email":
        edge_index_tensor = torch.tensor(np.loadtxt('datasets/EML_full.txt', dtype=np.int64).T, dtype=torch.long) - 1
    else:
        raise Exception("Unknown dataset: ", args.dataset)
    
    data = Data(edge_index=edge_index_tensor)
    data.num_nodes = edge_index_tensor.max().item() + 1
    data.x = torch.ones(data.num_nodes, 1)

    transform = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=True),
    ])

    train_data, val_data, test_data = transform(data)

    print("random sampling to determine K in DGCNN")
    sample_idx = torch.randperm(train_data.edge_label_index.size(1))[:1000]
    sampled_edge_label_index = train_data.edge_label_index[:, sample_idx]
    sampled_data = extract_enclosing_subgraphs(train_data.x, train_data.edge_index, sampled_edge_label_index, 2, proc_bar=True)
    num_nodes = sorted([g.num_nodes for g in sampled_data])
    k = num_nodes[int(math.ceil(args.k_ratio * len(num_nodes))) - 1]
    k = max(10, k)

    model = DGCNN(num_features=args.z_max + 1, hidden_channels=32, k=k).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()

    print(model)
    with open(f'seal_training_log_{args.dataset}.txt', 'a') as file:
        best_val_auc = test_auc = 0
        for epoch in range(1, args.epochs + 1):
            loss = train(model, optimizer, train_data, args.batch_size, criterion, device, args.z_max)
            val_auc = test(model, val_data, args.batch_size, device, args.z_max)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                test_auc = test(model, test_data, args.batch_size, device, args.z_max)
            log = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}'
            print(log)
            file.write(log + '\n')
            file.flush()

def train(model, optimizer, train_data, batch_size, criterion, device, z_max):
    model.train()
    total_loss = 0
    for perm in tqdm(DataLoader(range(train_data.edge_label_index.size(1)), batch_size, shuffle=True)):
        sampled_edge_label_index = train_data.edge_label_index[:, perm]
        sampled_data = extract_enclosing_subgraphs(train_data.x, train_data.edge_index, sampled_edge_label_index, 2)
        sampled_data = Batch.from_data_list(sampled_data)
        sampled_data.x = F.one_hot(torch.clamp(sampled_data.z, max=z_max), z_max + 1).to(device).float()
        optimizer.zero_grad()
        out = model(sampled_data.x, sampled_data.edge_index, sampled_data.batch)
        loss = criterion(out.view(-1), train_data.edge_label[perm])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss) * sampled_data.num_graphs
    return total_loss / train_data.edge_label_index.size(1)


@torch.no_grad()
def test(model, test_data, batch_size, device, z_max):
    model.eval()
    y_pred, y_true = [], []
    for perm in tqdm(DataLoader(range(test_data.edge_label_index.size(1)), batch_size)):
        edge_label_index = test_data.edge_label_index[:, perm]
        batch_data = extract_enclosing_subgraphs(test_data.x, test_data.edge_index, edge_label_index, 2)
        batch_data = Batch.from_data_list(batch_data)
        batch_data.x = F.one_hot(torch.clamp(batch_data.z, max=z_max), z_max + 1).to(device).float()
        logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)                
        y_pred.append(logits.view(-1).cpu())
        y_true.append(test_data.edge_label[perm].view(-1).cpu().to(torch.float))
    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred))

if __name__ == "__main__":
    main()
