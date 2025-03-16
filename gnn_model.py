import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# class GNNModel(nn.Module):
    # def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
    #     super(GNNModel, self).__init__()
    #     self.conv1 = GCNConv(input_dim, hidden_dim)
    #     self.conv2 = GCNConv(hidden_dim, hidden_dim)
    #     self.conv3 = GCNConv(hidden_dim, hidden_dim)
    #     self.fc = nn.Linear(hidden_dim, output_dim)

    # def forward(self, x, edge_index, batch):
    #     x = self.conv1(x, edge_index).relu()
    #     x = self.conv2(x, edge_index).relu()
    #     x = global_mean_pool(x, batch)
    #     return self.fc(x)

class GNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x, edge_index, batch):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Fully connected layer
        x = self.fc(x)

        # Optional: Apply activation function if needed
        # x = F.relu(x)  # For non-negative targets
        # x = torch.sigmoid(x)  # For targets in [0, 1]
        return x