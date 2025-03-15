from torch_geometric.datasets import QM9
from constants import QM9_PROPS_UNITS


# Load the QM9 dataset (without transform for raw data)
dataset = QM9(root='data/QM9')

# Select a sample molecule (first molecule in the dataset)
data = dataset[0]

# Print basic information
print("Molecular Graph Information:")
print(f"Number of atoms (nodes): {data.x.shape[0]}")
print(f"Number of bonds (edges): {data.edge_index.shape[1]}")

# Print atomic features (first few atoms)
print("\nAtom Features (First 5 Atoms):")
print(data.x[:5])

# Print bond information (first few edges)
print("\nBond Indices (First 5 Bonds):")
print(data.edge_index[:, :5])

# Print molecular properties
print("\nMolecular Properties:")
for indx, property in enumerate(QM9_PROPS_UNITS):
    print(f"{property[0]}: {data.y[0, indx].item()}")
