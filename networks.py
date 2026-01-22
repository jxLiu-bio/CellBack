import torch
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn as nn
import torch.nn.functional as F


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=4000, hidden=500, heads=8, dropout=0.2):
        super(GATEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(int(hidden * heads), int(out_channels / heads), heads=heads, concat=True, dropout=dropout)

    def forward(self, x, edge_index, p=0.2):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=4000, hidden=500, heads=8, dropout=0.2):
        super(GATDecoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(int(hidden * heads), int(out_channels / heads), heads=heads, concat=True, dropout=dropout)

    def forward(self, x, edge_index, p=0.2):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=self.training)
        x = self.conv2(x, edge_index)
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