import numpy as np
import scanpy as sc
import torch
import seaborn as sns
import torch.nn as nn
from cellback.networks import GraphAutoencoder, GATEncoder, GATDecoder
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, dense_to_sparse
from torch_geometric.data import Data


def micro_network_training(adata_sp,
                           adata_sc,
                           features,
                           edge_index,
                           targets,
                           device='auto',
                           learning_rate=0.0002,
                           epoch=30000,
                           hidden_dim=500,
                           heads=8,
                           dropout=0.2,
                           cell_types=None,
                           weight=5,
                           ):
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data format transform
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)
    data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index)

    # adjust
    input_features_num_1 = features.shape[1]
    output_features_num_1 = targets.shape[1]
    input_features_num_2 = targets.shape[1]
    output_features_num_2 = features.shape[1]
    encoder = GATEncoder(input_features_num_1, output_features_num_1, hidden=hidden_dim,
                         heads=heads, dropout=dropout).to(device)
    decoder = GATDecoder(input_features_num_2, output_features_num_2, hidden=hidden_dim,
                         heads=heads, dropout=dropout).to(device)
    model = GraphAutoencoder(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    data = data.to(device)
    exp_sp = torch.tensor(features[:targets.shape[0], :], dtype=torch.float).to(device)
    exp_sc = torch.tensor(features[targets.shape[0]:, :], dtype=torch.float).to(device)
    targets = torch.tensor(targets, dtype=torch.float).to(device)

    def train(data, epochs=epoch):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            x_hat, z = model(data.x, data.edge_index)
            samples = cell_types[epoch % len(cell_types)]

            names_sp = adata_sp.obs[adata_sp.obs['cell_type'] == samples].index
            cell_indices_sp = adata_sp.obs.index.get_indexer(names_sp)
            names_sc = adata_sc.obs[adata_sc.obs['cell_type'] == samples].index
            cell_indices_sc = adata_sc.obs.index.get_indexer(names_sc)

            true_mean = torch.mean(targets[cell_indices_sp, :], dim=0)
            true_var = torch.var(targets[cell_indices_sp, :], dim=0, unbiased=False)
            slide_mean = torch.mean(z[cell_indices_sp, :], dim=0)
            slide_var = torch.var(z[cell_indices_sp, :], dim=0, unbiased=False)
            indices = [exp_sp.shape[0] + idx for idx in cell_indices_sc]
            sc_mean = torch.mean(z[indices, :], dim=0)
            sc_var = torch.var(z[indices, :], dim=0, unbiased=False)

            loss1 = criterion(true_mean, slide_mean) + criterion(true_var, slide_var) + \
                    criterion(true_mean, sc_mean) + weight * criterion(true_var, sc_var)

            true_mean_all = torch.mean(targets[:, :], dim=0)
            true_var_all = torch.var(targets[:, :], dim=0, unbiased=False)
            slide_mean_all = torch.mean(z[0:exp_sp.shape[0], :], dim=0)
            slide_var_all = torch.var(z[0:exp_sp.shape[0], :], dim=0, unbiased=False)
            sc_mean_all = torch.mean(z[exp_sp.shape[0]:features.shape[0], :], dim=0)
            sc_var_all = torch.var(z[exp_sp.shape[0]:features.shape[0], :], dim=0, unbiased=False)

            loss2 = criterion(true_mean_all, slide_mean_all) + criterion(true_var_all, slide_var_all) + criterion(
                true_mean_all, sc_mean_all) + weight * criterion(true_var_all, sc_var_all)

            loss3 = 2 * criterion(z[:exp_sp.shape[0], :], targets) + \
                    2 * criterion(x_hat[:exp_sp.shape[0], :], exp_sp) + \
                    4 * criterion(x_hat[exp_sp.shape[0]:, :], exp_sc)

            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    train(data)

    model.eval()
    _, micro_embeddings = model(data.x, data.edge_index)
    micro_embeddings = micro_embeddings.cpu().detach().numpy()

    # add micro embedding to anndata objects
    adata_sc.obsm['embedding'] = np.hstack((exp_sc.cpu().detach().numpy(),
                                            micro_embeddings[exp_sp.shape[0]:features.shape[0],
                                                    :]))
    adata_sp.obsm['embedding'] = np.hstack((exp_sp.cpu().detach().numpy(),
                                            micro_embeddings[:exp_sp.shape[0], :]))


