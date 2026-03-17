from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader


def evaluate_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[float, float]:
    model.eval()
    test_loss, test_mae = 0.0, 0.0

    for batch in tqdm(test_dataloader):
        batch = batch.to(device)
        with torch.inference_mode():
            preds = model(batch.x.float(), batch.edge_index, batch.batch)

        loss = loss_fn(preds, batch.y)
        test_loss += loss.item()
        test_mae += torch.mean(torch.abs(preds - batch.y)).item()

    test_loss /= len(test_dataloader)
    test_mae /= len(test_dataloader)

    print(f"Test Loss (MSE): {test_loss:.4f} | Test MAE: {test_mae:.4f} mol/L")

    return test_loss, test_mae
