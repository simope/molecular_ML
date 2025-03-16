import torch
from train import train
from evaluate import evaluate


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train()
# evaluate()
