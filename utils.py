import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix, dropout_path
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_cluster import graclus_cluster
from torch_geometric.nn import max_pool


def extract_enclosing_subgraphs(x, edge_index, edge_label_index, num_hops, proc_bar=False):
    def process_pair(src_dst):
        src, dst = src_dst
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops, edge_index, relabel_nodes=True, num_nodes=x.size(0))
        src, dst = mapping.tolist()

        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))

        return Data(x=x[sub_nodes], z=z, edge_index=sub_edge_index)


    data_list = []
    edge_list = edge_label_index.t().tolist()
    
    if proc_bar:
        pbar = tqdm(total=len(edge_list))

    for pair in edge_list:
        data_list.append(process_pair(pair))            
        if proc_bar:
            pbar.update()
    if proc_bar:
        pbar.close()

    return data_list


def extract_enclosing_subgraphs_with_cluster_label(x, edge_index, edge_label_index, num_hops, proc_bar=False, resolution=5):
    def process_pair(src_dst):
        src, dst = src_dst
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops, edge_index, relabel_nodes=True, num_nodes=x.size(0))
        src, dst = mapping.tolist()
                
        mask1 = (sub_edge_index[0] != src) & (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) & (sub_edge_index[1] != src)
        cluster_edge_index = sub_edge_index[:, mask1 & mask2]
        cluster = graclus_cluster(cluster_edge_index[0], cluster_edge_index[1], num_nodes=sub_nodes.size(0))
        data = Data(edge_index=sub_edge_index)
        pooled_data = max_pool(cluster, data)
        unique, inv = torch.unique(cluster, sorted=True, return_inverse=True)

        z_prime = drnl_node_labeling(pooled_data.edge_index, inv[src].item(), inv[dst].item(), num_nodes=unique.size(0))
        z_prime = z_prime[inv.cpu()]    

        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]
        z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))


        return Data(x=x[sub_nodes], z=z, z_prime=z_prime, edge_index=sub_edge_index)


    data_list = []
    edge_list = edge_label_index.t().tolist()
    
    if proc_bar:
        pbar = tqdm(total=len(edge_list))

    for pair in edge_list:
        data_list.append(process_pair(pair))            
        if proc_bar:
            pbar.update()
    if proc_bar:
        pbar.close()

    return data_list


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    # Double-radius node labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)
   
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    
    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                                indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                                indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)

