import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def create_graph_data(atom_features, bond_indices, bond_features, target):
    return Data(
        x=torch.tensor(atom_features, dtype=torch.float32),
        edge_index=torch.tensor(bond_indices, dtype=torch.long),
        edge_attr=torch.tensor(bond_features, dtype=torch.float32),
        y=torch.tensor(target, dtype=torch.float32),
    )
