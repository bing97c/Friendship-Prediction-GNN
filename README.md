# Friendship-Prediction-GNN
Friendship Prediction Based on Graph Neural Networks

## Setup:

```
torch==2.1.2
torch_geometric==2.5.2
```
Install pyg requirements:

```
pip install -q torch-geometric
pip install -q torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2+cu118.html
```

## Usage Example

```
python seal.py --dataset brightkite
python seal_cluster.py --dataset brightkite
python vgae.py --dataset brightkite
```
