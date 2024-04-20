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
from models import DGCNN, VariationalGCNEncoder
from torch_geometric.nn import VGAE
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
import random

# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

class Encoder(nn.Module):
    def __init__(self, in_features, out_channels):
        super().__init__()
        self.vgae = VGAE(VariationalGCNEncoder(64, out_channels))
        self.emb = nn.Embedding(in_features, 64)
    def encode(self, edge_index):
        z = self.vgae.encode(self.emb.weight, edge_index)
        return z
    def recon_loss(self, z, edge_label_index):
        return self.vgae.recon_loss(z, edge_label_index)
    def test(self, z, pos_edge_label_index, neg_edge_label_index):
        return self.vgae.test(z, pos_edge_label_index, neg_edge_label_index)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--k_ratio', type=float, default=0.6)
    parser.add_argument('--z_max', type=int, default=200)
    parser.add_argument('--num_encoder_features', type=int, default=64)
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

    encoder = Encoder(data.num_nodes, args.num_encoder_features).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), args.lr)

    train_pos_edge_label_index = train_data.edge_label_index[:, (train_data.edge_label == 1).nonzero().flatten()]
    val_pos_edge_label_index = val_data.edge_label_index[:, (val_data.edge_label == 1).nonzero().flatten()]
    val_neg_edge_label_index = val_data.edge_label_index[:, (val_data.edge_label == 0).nonzero().flatten()]   
    best_val_auc = test_auc = 0
    for epoch in range(1, 200 + 1):
        encoder.train()
        optimizer.zero_grad()
        z = encoder.encode(train_data.edge_index)
        loss = encoder.recon_loss(z, train_pos_edge_label_index)
        loss = loss + (1 / train_data.num_nodes) * encoder.vgae.kl_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
        optimizer.step()

        with torch.no_grad():
            encoder.eval()
            z = encoder.encode(val_data.edge_index)
            val_auc = encoder.test(z, val_pos_edge_label_index, val_neg_edge_label_index)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            trained_encoder = copy.deepcopy(encoder)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}', f'Best: {best_val_auc:.4f}')

    with torch.no_grad():
        train_data.x = trained_encoder.encode(train_data.edge_index)
        val_data.x = trained_encoder.encode(val_data.edge_index)
        test_data.x = trained_encoder.encode(test_data.edge_index)

    print("random sampling to determine K in DGCNN")
    sample_idx = torch.randperm(train_data.edge_label_index.size(1))[:1000]
    sampled_edge_label_index = train_data.edge_label_index[:, sample_idx]
    sampled_data = extract_enclosing_subgraphs(train_data.x, train_data.edge_index, sampled_edge_label_index, 2, proc_bar=True)
    num_nodes = sorted([g.num_nodes for g in sampled_data])
    k = num_nodes[int(math.ceil(args.k_ratio * len(num_nodes))) - 1]
    k = max(10, k)

    model = DGCNN(num_features=args.z_max + 1 + args.num_encoder_features, hidden_channels=32, k=k).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()

    print(model)
    with open(f'seal_gvae_training_log_{args.dataset}.txt', 'a') as file:
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
        sampled_data.x = torch.cat([
            sampled_data.x,
            F.one_hot(torch.clamp(sampled_data.z, max=z_max), z_max + 1).to(device).float(),
        ], dim=-1)        
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
        batch_data.x = torch.cat([
            batch_data.x,
            F.one_hot(torch.clamp(batch_data.z, max=z_max), z_max + 1).to(device).float()
        ], dim=-1)             
        logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)                
        y_pred.append(logits.view(-1).cpu())
        y_true.append(test_data.edge_label[perm].view(-1).cpu().to(torch.float))
    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred))

if __name__ == "__main__":
    main()
