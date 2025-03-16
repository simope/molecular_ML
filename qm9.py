from torch_geometric.datasets import QM9

def load_qm9(root='data/QM9'):
    dataset = QM9(root=root)
    return dataset