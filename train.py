import torch
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

from qm9 import load_qm9
from gnn_model import GNNModel


def train():
    dataset = load_qm9()

    # Define which targets to predict
    targets_indx = [2] # HOMO, LUMO

    # Precompute mean and std for the entire dataset
    all_targets = torch.cat([data.y[:, targets_indx] for data in dataset], dim=0)
    mean = all_targets.mean(dim=0)
    std = all_targets.std(dim=0)

    # Initialize DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer and loss function
    learning_rate = 0.05
    model = GNNModel(input_dim=11, hidden_dim=512, output_dim=len(targets_indx))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Training loop
    num_epochs = 50
    with open("logs/training_log.log", "a") as log_file:
        msg = f"Starting training...\nLearning rate: {learning_rate}"
        tqdm.write(msg, file=log_file)
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for data in progress_bar:
                optimizer.zero_grad()
                # Normalization of target values
                norm_targets = (data.y[:, targets_indx] - mean) / std

                # Forward pass
                norm_out = model(data.x, data.edge_index, data.batch)

                # Compute loss
                loss = criterion(norm_out, norm_targets)

                # Backward pass
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
                # Update tqdm progress bar with current loss
                progress_bar.set_postfix(loss=loss.item())

            msg = f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss/len(train_loader):.4f}"
            print(msg)
            tqdm.write(msg, file=log_file)

    # Save the trained model
    name = "gnn_model.pth"
    torch.save(model.state_dict(), name)
    print(f"Model saved as '{name}'")