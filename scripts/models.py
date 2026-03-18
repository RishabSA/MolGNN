import torch
import torch.nn as nn
from torch_geometric.nn import (
    Sequential,
    GCNConv,
    GATConv,
    GATv2Conv,
    GINConv,
    TransformerConv,
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

        self.layers = nn.ModuleList(
            [
                Sequential(
                    "x, edge_index",
                    [
                        (
                            GCNConv(
                                in_channels=hidden_channels,
                                out_channels=hidden_channels,
                            ),
                            "x, edge_index -> x",
                        ),
                        (BatchNorm(in_channels=hidden_channels), "x -> x"),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                    ],
                )
                for _ in range(num_layers)
            ]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = x + layer(x, edge_index)

        return global_mean_pool(x, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = x + layer(x, edge_index)

        x = global_mean_pool(x, batch)

        return self.output_head(x)


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
                        concat=False,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                (BatchNorm(in_channels=hidden_channels), "x -> x"),
                nn.ELU(),
                nn.Dropout(p=dropout),
            ],
        )

        self.layers = nn.ModuleList(
            [
                Sequential(
                    "x, edge_index",
                    [
                        (
                            GATConv(
                                in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                heads=heads,
                                concat=False,
                                dropout=dropout,
                            ),
                            "x, edge_index -> x",
                        ),
                        (BatchNorm(in_channels=hidden_channels), "x -> x"),
                        nn.ELU(),
                        nn.Dropout(p=dropout),
                    ],
                )
                for _ in range(num_layers)
            ]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = x + layer(x, edge_index)

        return global_mean_pool(x, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = x + layer(x, edge_index)

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

        self.layers = nn.ModuleList(
            [
                Sequential(
                    "x, edge_index",
                    [
                        (
                            GINConv(
                                nn=nn.Sequential(
                                    nn.Linear(hidden_channels, hidden_channels),
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
                for _ in range(num_layers)
            ]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = x + layer(x, edge_index)

        return global_mean_pool(x, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index)

        for layer in self.layers:
            x = x + layer(x, edge_index)

        x = global_mean_pool(x, batch)

        return self.output_head(x)


class GraphAttentionV2Network(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        in_channels: int = 9,
        hidden_channels: int = 64,
        out_channels: int = 1,
        heads: int = 4,
        edge_dim: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_head = Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    GATv2Conv(
                        in_channels=in_channels,
                        out_channels=hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    ),
                    "x, edge_index, edge_attr -> x",
                ),
                (BatchNorm(in_channels=hidden_channels), "x -> x"),
                nn.ELU(),
                nn.Dropout(p=dropout),
            ],
        )

        self.layers = nn.ModuleList(
            [
                Sequential(
                    "x, edge_index, edge_attr",
                    [
                        (
                            GATv2Conv(
                                in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                heads=heads,
                                concat=False,
                                dropout=dropout,
                                edge_dim=edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        ),
                        (BatchNorm(in_channels=hidden_channels), "x -> x"),
                        nn.ELU(),
                        nn.Dropout(p=dropout),
                    ],
                )
                for _ in range(num_layers)
            ]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index, edge_attr)

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)

        return global_mean_pool(x, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index, edge_attr)

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        return self.output_head(x)


class GraphTransformerNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        in_channels: int = 9,
        hidden_channels: int = 64,
        out_channels: int = 1,
        heads: int = 4,
        edge_dim: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_head = Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    TransformerConv(
                        in_channels=in_channels,
                        out_channels=hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    ),
                    "x, edge_index, edge_attr -> x",
                ),
                (BatchNorm(in_channels=hidden_channels), "x -> x"),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ],
        )

        self.layers = nn.ModuleList(
            [
                Sequential(
                    "x, edge_index, edge_attr",
                    [
                        (
                            TransformerConv(
                                in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                heads=heads,
                                concat=False,
                                dropout=dropout,
                                edge_dim=edge_dim,
                                beta=True,
                            ),
                            "x, edge_index, edge_attr -> x",
                        ),
                        (BatchNorm(in_channels=hidden_channels), "x -> x"),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                    ],
                )
                for _ in range(num_layers)
            ]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index, edge_attr)

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)

        return global_mean_pool(x, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_head(x, edge_index, edge_attr)

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)

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
