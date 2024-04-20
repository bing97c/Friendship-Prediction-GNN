import argparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn import VGAE
from models import VariationalGCNEncoder
import numpy as np
import torch
from torch_geometric.data import Data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variational', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)    
    parser.add_argument('--dataset', type=str, default='brightkite')
    parser.add_argument('--epochs', type=int, default=200)

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
                        split_labels=True, add_negative_train_samples=False),
    ])

    train_data, val_data, test_data = transform(data)

    model = VGAE(VariationalGCNEncoder(64, 64)).to(device)
    emb = nn.Embedding(data.x.size(0), 64).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(emb.parameters()), lr=args.lr)
    nn.init.xavier_uniform_(emb.weight)
    
    best_val_auc = test_auc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, emb.weight, optimizer, train_data)
        val_auc = test(model, emb.weight, val_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = test(model, emb.weight, test_data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
            f'Test: {test_auc:.4f}')


def train(model, x, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    torch.nn.utils.clip_grad_norm_(x, 1)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    return float(loss)

@torch.no_grad()
def test(model, x, data):
    model.eval()
    z = model.encode(x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)[0]

if __name__ == "__main__":
    main()
