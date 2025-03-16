import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        
        # First Graph Convolution Layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Second Graph Convolution Layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Apply first convolution
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # Apply second convolution
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Global mean pooling to aggregate node features
        x = global_mean_pool(x, batch)
        
        # Output prediction
        x = self.fc(x)
        return x
