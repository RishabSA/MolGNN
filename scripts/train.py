import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int = 200,
    save_path: str = "models/model.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[nn.Module, list, list, list]:
    progress_bar = tqdm(range(num_epochs), desc=f"Training for {num_epochs} epochs")
    train_losses, test_losses, test_maes = [], [], []

    for epoch in progress_bar:
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_attr = batch.edge_attr.float() if batch.edge_attr is not None else None
            preds = model(batch.x.float(), batch.edge_index, batch.batch, edge_attr)
            loss = loss_fn(preds, batch.y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        if epoch % 10 == 0:
            model.eval()
            test_loss, test_mae = 0.0, 0.0

            for batch in test_dataloader:
                batch = batch.to(device)
                with torch.inference_mode():
                    edge_attr = batch.edge_attr.float() if batch.edge_attr is not None else None
                    preds = model(batch.x.float(), batch.edge_index, batch.batch, edge_attr)

                loss = loss_fn(preds, batch.y)
                test_loss += loss.item()
                test_mae += torch.mean(torch.abs(preds - batch.y)).item()

            test_loss /= len(test_dataloader)
            test_mae /= len(test_dataloader)
            test_losses.append(test_loss)
            test_maes.append(test_mae)

            print(
                f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f} mol/L"
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, train_losses, test_losses, test_maes
