import torch
import torch.nn as nn
from torch_geometric.nn import (
    Sequential,
    GCNConv,
    GATConv,
    GINConv,
    BatchNorm,
    global_mean_pool,
)


class GraphConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        in_channels: int = 9,
        hidden_channels: int = 64,
        out_channels: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_head = Sequential(
            "x, edge_index",
            [
                (
                    GCNConv(in_channels=in_channels, out_channels=hidden_channels),
                    "x, edge_index -> x",
                ),
                (BatchNorm(in_channels=hidden_channels), "x -> x"),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ],
        )

        self.layers = nn.ModuleList()

        initial_in_channels = hidden_channels

        for i in range(num_layers):
            self.layers.append(
                Sequential(
                    "x, edge_index",
                    [
                        (
                            GCNConv(
                                in_channels=initial_in_channels,
                                out_channels=initial_in_channels // 2,
                            ),
                            "x, edge_index -> x",
                        ),
                        (
                            BatchNorm(in_channels=initial_in_channels // 2),
                            "x -> x",
                        ),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                    ],
                )
            )

            initial_in_channels //= 2

        self.output_head = nn.Linear(
            in_features=initial_in_channels, out_features=out_channels
        )

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        # x shape: (N, in_channels)
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        return global_mean_pool(x, batch)  # shape: (N, final_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        # x shape: (N, in_channels)
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = global_mean_pool(x, batch)  # shape: (N, final_channels)

        return self.output_head(x)  # shape: (N, out_channels)


class GraphAttentionNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        in_channels: int = 9,
        hidden_channels: int = 64,
        out_channels: int = 1,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_head = Sequential(
            "x, edge_index",
            [
                (
                    GATConv(
                        in_channels=in_channels,
                        out_channels=hidden_channels,
                        heads=heads,
                        concat=False,  # average heads so output dim = hidden_channels
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                (BatchNorm(in_channels=hidden_channels), "x -> x"),
                nn.ELU(),
                nn.Dropout(p=dropout),
            ],
        )

        self.layers = nn.ModuleList()

        initial_in_channels = hidden_channels

        for i in range(num_layers):
            self.layers.append(
                Sequential(
                    "x, edge_index",
                    [
                        (
                            GATConv(
                                in_channels=initial_in_channels,
                                out_channels=initial_in_channels // 2,
                                heads=heads,
                                concat=False,
                                dropout=dropout,
                            ),
                            "x, edge_index -> x",
                        ),
                        (
                            BatchNorm(in_channels=initial_in_channels // 2),
                            "x -> x",
                        ),
                        nn.ELU(),
                        nn.Dropout(p=dropout),
                    ],
                )
            )

            initial_in_channels //= 2

        self.output_head = nn.Linear(
            in_features=initial_in_channels, out_features=out_channels
        )

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        return global_mean_pool(x, batch)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = global_mean_pool(x, batch)

        return self.output_head(x)


class GraphIsomorphismNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        in_channels: int = 9,
        hidden_channels: int = 64,
        out_channels: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_head = Sequential(
            "x, edge_index",
            [
                (
                    GINConv(
                        nn=nn.Sequential(
                            nn.Linear(in_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels),
                        ),
                        train_eps=True,
                    ),
                    "x, edge_index -> x",
                ),
                (BatchNorm(in_channels=hidden_channels), "x -> x"),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ],
        )

        self.layers = nn.ModuleList()

        initial_in_channels = hidden_channels

        for i in range(num_layers):
            self.layers.append(
                Sequential(
                    "x, edge_index",
                    [
                        (
                            GINConv(
                                nn=nn.Sequential(
                                    nn.Linear(
                                        initial_in_channels, initial_in_channels // 2
                                    ),
                                    nn.ReLU(),
                                    nn.Linear(
                                        initial_in_channels // 2,
                                        initial_in_channels // 2,
                                    ),
                                ),
                                train_eps=True,
                            ),
                            "x, edge_index -> x",
                        ),
                        (
                            BatchNorm(in_channels=initial_in_channels // 2),
                            "x -> x",
                        ),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                    ],
                )
            )

            initial_in_channels //= 2

        self.output_head = nn.Linear(
            in_features=initial_in_channels, out_features=out_channels
        )

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        return global_mean_pool(x, batch)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = global_mean_pool(x, batch)

        return self.output_head(x)


def load_model(
    model: nn.Module,
    path: str = "models/model.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> nn.Module:

    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model
