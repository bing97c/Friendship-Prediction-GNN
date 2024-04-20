import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MLP, SortAggregation


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            GCNConv(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            GCNConv(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = MLP([hidden_channels, hidden_channels, out_channels])
    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

    def decode(self, z):
        return self.mlp(z)

class DGCNN(nn.Module):
    def __init__(self, num_features, hidden_channels, k):
        super().__init__()
        self.gcn = nn.Sequential(
            GCNConv(num_features, hidden_channels),
            nn.Tanh(),
            GCNConv(hidden_channels, hidden_channels),
            nn.Tanh(),
            GCNConv(hidden_channels, 1),
            nn.Tanh()
        )
        
        self.pool = SortAggregation(k)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, hidden_channels * 2 + 1, hidden_channels * 2 + 1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 5, 1),
            nn.ReLU(inplace=True)
        )
        self.mlp_in_dim = (int((k - 2) / 2 + 1) - 4) * hidden_channels
        self.hidden_channels = hidden_channels
        self.mlp = MLP([self.mlp_in_dim, 128, 1])

    def aggregate(self, x, edge_index, batch):
        xs = []
        for layer in self.gcn:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
                xs.append(x)
        x = torch.cat(xs, dim=-1)
        x = self.pool(x, batch).unsqueeze(1)
        return x
    
    def decode(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

    def decode2(self, x):
        x = self.cnn(x)
        return x


    def forward(self, x, edge_index, batch):
        return self.decode(self.aggregate(x, edge_index, batch))


class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

