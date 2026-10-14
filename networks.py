import torch
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=4000,
        hidden=500,
        heads=8,
        dropout=0.2,
        num_layers=2
    ):
        super(GATEncoder, self).__init__()

        if num_layers < 2:
            raise ValueError("num_layers 必须大于等于 2")

        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels={out_channels} 必须能被 heads={heads} 整除"
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            GATv2Conv(
                in_channels,
                hidden,
                heads=heads,
                dropout=dropout
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden * heads,
                    hidden,
                    heads=heads,
                    dropout=dropout
                )
            )

        self.convs.append(
            GATv2Conv(
                hidden * heads,
                out_channels // heads,
                heads=heads,
                concat=True,
                dropout=dropout
            )
        )

    def forward(self, x, edge_index, p=0.2):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(
                x,
                p=p,
                training=self.training
            )

        x = self.convs[-1](x, edge_index)

        return x


class GATDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=4000,
        hidden=500,
        heads=8,
        dropout=0.2,
        num_layers=2
    ):
        super(GATDecoder, self).__init__()

        if num_layers < 2:
            raise ValueError("num_layers 必须大于等于 2")

        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels={out_channels} 必须能被 heads={heads} 整除"
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            GATv2Conv(
                in_channels,
                hidden,
                heads=heads,
                dropout=dropout
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden * heads,
                    hidden,
                    heads=heads,
                    dropout=dropout
                )
            )

        self.convs.append(
            GATv2Conv(
                hidden * heads,
                out_channels // heads,
                heads=heads,
                concat=True,
                dropout=dropout
            )
        )

    def forward(self, x, edge_index, p=0.2):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(
                x,
                p=p,
                training=self.training
            )

        x = self.convs[-1](x, edge_index)

        return x


class GraphAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GraphAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z, edge_index)

        return x_hat, z