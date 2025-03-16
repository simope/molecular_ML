import torch
from torch_geometric.loader import DataLoader

from gnn_model import GNNModel
from qm9 import load_qm9


def evaluate():
    dataset = load_qm9()
    test_loader = DataLoader(dataset, batch_size=32)

    model = GNNModel(input_dim=11, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load("gnn_model.pth"))
    model.eval()

    predictions, targets = [], []
    with torch.no_grad():
        for data in test_loader:
            pred = model(data.x, data.edge_index, data.batch)
            predictions.append(pred)
            targets.append(data.y)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    rmse = torch.sqrt(((predictions - targets) ** 2).mean())
    print(f"Test RMSE: {rmse:.4f}")
