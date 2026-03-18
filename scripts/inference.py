import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles


def smiles_to_data(smiles: str, y: float = None) -> Data:
    data = from_smiles(smiles)

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data


def predict_log_solubility_from_smiles(
    model: nn.Module,
    smiles_seq: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    model.eval()

    data = smiles_to_data(smiles=smiles_seq).to(device)
    # All nodes belong to graph 0 becuase it is a single graph
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    with torch.inference_mode():
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        log_sol = model(data.x.float(), data.edge_index, batch, edge_attr)

    print(f"Predicted Log Solubility: {log_sol.item():.4f} mol/L")

    return log_sol.item()
