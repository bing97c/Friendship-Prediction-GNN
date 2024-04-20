import math
import argparse
from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from utils import extract_enclosing_subgraphs  
from torch_geometric.nn import Node2Vec, MLP
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--embedding_training_epochs', type=int, default=50)    
    parser.add_argument('--dataset', type=str, default='email')
        
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

    emb = Node2Vec(train_data.edge_index, args.embedding_dim, args.walk_length,
                    args.context_size, args.walks_per_node, num_nodes=data.num_nodes,
                    sparse=True).to(device)

    loader = emb.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    optimizer = torch.optim.SparseAdam(list(emb.parameters()), lr=args.lr)


    emb.train()
    for epoch in range(1, args.embedding_training_epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = emb.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                        f'Loss: {loss:.4f}')

    x = emb.embedding.weight.clone().detach()

    del emb

    model = MLP([args.embedding_dim, 256, 256, 1]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()

    print(model)
    with open(f'training_log_{args.dataset}.txt', 'a') as file:
        best_val_auc = test_auc = 0
        for epoch in range(1, args.epochs + 1):
            loss = train(model, x, optimizer, train_data, args.batch_size, criterion)
            val_auc = test(model, x, val_data, args.batch_size)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                test_auc = test(model, x, test_data, args.batch_size)
            log = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}'
            print(log)
            file.write(log + '\n')
            file.flush()

def train(model, x, optimizer, train_data, batch_size, criterion):
    model.train()
    total_loss = 0
    for perm in tqdm(DataLoader(range(train_data.edge_label_index.size(1)), batch_size, shuffle=True)):
        sampled_edge_label_index = train_data.edge_label_index[:, perm]
        
        optimizer.zero_grad()
        out = model(x[sampled_edge_label_index[0]] * x[sampled_edge_label_index[1]])
        loss = criterion(out.view(-1), train_data.edge_label[perm])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss) * perm.size(0)
    return total_loss / train_data.edge_label_index.size(1)

@torch.no_grad()
def test(model, x, test_data, batch_size):
    model.eval()
    y_pred, y_true = [], []
    for perm in tqdm(DataLoader(range(test_data.edge_label_index.size(1)), batch_size)):
        edge_label_index = test_data.edge_label_index[:, perm]           
        logits = model(x[edge_label_index[0]] * x[edge_label_index[1]] )                
        y_pred.append(logits.view(-1).cpu())
        y_true.append(test_data.edge_label[perm].view(-1).cpu().to(torch.float))
    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred))


if __name__ == "__main__":
    main()