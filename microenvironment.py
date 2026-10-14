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
                           num_layers=2,
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
                         heads=heads, dropout=dropout, num_layers=num_layers).to(device)
    decoder = GATDecoder(input_features_num_2, output_features_num_2, hidden=hidden_dim,
                         heads=heads, dropout=dropout, num_layers=num_layers).to(device)
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
            if epoch % 100 == 0:
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



def build_grid_subgraphs(
        adata_sp,
        adata_sc,
        full_edge_index,
        k=10,
        spatial_key='spatial'
):

    n_sp = adata_sp.n_obs
    n_sc = adata_sc.n_obs
    n_nodes = n_sp + n_sc

    if not torch.is_tensor(full_edge_index):
        full_edge_index = torch.as_tensor(
            full_edge_index,
            dtype=torch.long
        )

    coordinates = np.asarray(
        adata_sp.obsm[spatial_key]
    )

    coordinates = coordinates[:, :2].astype(np.float64)

    x_coordinate = coordinates[:, 0]
    y_coordinate = coordinates[:, 1]

    def assign_grid_index(values, grid_number):
        value_min = np.min(values)
        value_max = np.max(values)

        if np.isclose(value_min, value_max):
            return np.zeros(
                len(values),
                dtype=np.int64
            )

        boundaries = np.linspace(
            value_min,
            value_max,
            grid_number + 1
        )

        grid_index = np.searchsorted(
            boundaries[1:-1],
            values,
            side='right'
        )

        return np.clip(
            grid_index,
            0,
            grid_number - 1
        ).astype(np.int64)

    x_grid = assign_grid_index(x_coordinate, k)
    y_grid = assign_grid_index(y_coordinate, k)

    grid_ids = y_grid * k + x_grid

    full_edge_np = full_edge_index.numpy()

    source_nodes = full_edge_np[0]
    target_nodes = full_edge_np[1]

    sp_to_sc_mask = (
        (source_nodes < n_sp) &
        (target_nodes >= n_sp)
    )

    sc_to_sp_mask = (
        (source_nodes >= n_sp) &
        (target_nodes < n_sp)
    )

    cross_sp_nodes = np.concatenate([
        source_nodes[sp_to_sc_mask],
        target_nodes[sc_to_sp_mask]
    ]).astype(np.int64)

    cross_sc_nodes = np.concatenate([
        target_nodes[sp_to_sc_mask],
        source_nodes[sc_to_sp_mask]
    ]).astype(np.int64)

    if cross_sp_nodes.size == 0:
        raise ValueError(
            "edge_index中没有找到adata_sp和adata_sc之间的连边。"
        )

    sort_order = np.argsort(
        cross_sp_nodes,
        kind='stable'
    )

    cross_sp_nodes = cross_sp_nodes[sort_order]
    cross_sc_nodes = cross_sc_nodes[sort_order]

    neighbor_counts = np.bincount(
        cross_sp_nodes,
        minlength=n_sp
    )

    neighbor_indptr = np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(neighbor_counts)
    ])


    grid_subgraphs = []

    selected_node_mask = np.zeros(
        n_nodes,
        dtype=bool
    )

    for grid_id in range(k * k):

        sp_indices = np.flatnonzero(
            grid_ids == grid_id
        ).astype(np.int64)

        if sp_indices.size == 0:
            continue

        sc_neighbor_chunks = []

        for sp_index in sp_indices:
            start_position = neighbor_indptr[sp_index]
            end_position = neighbor_indptr[sp_index + 1]

            if start_position < end_position:
                sc_neighbor_chunks.append(
                    cross_sc_nodes[
                        start_position:end_position
                    ]
                )

        if len(sc_neighbor_chunks) == 0:
            continue

        sc_global_indices = np.unique(
            np.concatenate(sc_neighbor_chunks)
        ).astype(np.int64)

        sc_indices = sc_global_indices - n_sp

        node_ids = np.concatenate([
            sp_indices,
            sc_global_indices
        ]).astype(np.int64)

        selected_node_mask[node_ids] = True

        edge_mask = (
            selected_node_mask[source_nodes] &
            selected_node_mask[target_nodes]
        )

        sub_edge_global = full_edge_np[:, edge_mask]

        selected_node_mask[node_ids] = False

        if sub_edge_global.shape[1] == 0:
            continue

        sub_edge_local = np.searchsorted(
            node_ids,
            sub_edge_global
        ).astype(np.int64)

        grid_subgraphs.append({
            "grid_id": grid_id,
            "sp_indices": sp_indices,
            "sc_indices": sc_indices,
            "node_ids": torch.from_numpy(node_ids),
            "edge_index": torch.from_numpy(
                sub_edge_local
            ),

            "n_sp": len(sp_indices),
            "n_sc": len(sc_indices)
        })

    if len(grid_subgraphs) == 0:
        raise ValueError(
            "没有构建出有效子图，请检查空间坐标和edge_index。"
        )

    return grid_subgraphs



def calculate_subgraph_loss(
        model,
        subgraph,
        features_cpu,
        targets_cpu,
        device,
        criterion,
        weight=5,
):

    node_ids = subgraph["node_ids"]
    sub_edge_index = subgraph["edge_index"]
    sp_indices = subgraph["sp_indices"]
    n_sub_sp = subgraph["n_sp"]
    n_sub_sc = subgraph["n_sc"]


    sub_features = features_cpu.index_select(
        0,
        node_ids
    ).to(
        device,
        non_blocking=True
    )

    sub_edge_index = sub_edge_index.to(
        device,
        non_blocking=True
    )

    sp_indices_tensor = torch.as_tensor(
        sp_indices,
        dtype=torch.long
    )

    sub_targets = targets_cpu.index_select(
        0,
        sp_indices_tensor
    ).to(
        device,
        non_blocking=True
    )

    x_hat, z = model(
        sub_features,
        sub_edge_index
    )


    sub_exp_sp = sub_features[:n_sub_sp]

    sub_exp_sc = sub_features[
        n_sub_sp:n_sub_sp + n_sub_sc
    ]

    true_mean_all = torch.mean(
        sub_targets,
        dim=0
    )

    true_var_all = torch.var(
        sub_targets,
        dim=0,
        unbiased=False
    )

    slide_mean_all = torch.mean(
        z[:n_sub_sp],
        dim=0
    )

    slide_var_all = torch.var(
        z[:n_sub_sp],
        dim=0,
        unbiased=False
    )

    sc_mean_all = torch.mean(
        z[n_sub_sp:n_sub_sp + n_sub_sc],
        dim=0
    )

    sc_var_all = torch.var(
        z[n_sub_sp:n_sub_sp + n_sub_sc],
        dim=0,
        unbiased=False
    )

    distribution_loss = (
        criterion(
            true_mean_all,
            slide_mean_all
        )
        +
        criterion(
            true_var_all,
            slide_var_all
        )
        +
        criterion(
            true_mean_all,
            sc_mean_all
        )
        +
        weight * criterion(
            true_var_all,
            sc_var_all
        )
    )

    reconstruction_loss = (
        2 * criterion(
            z[:n_sub_sp],
            sub_targets
        )
        +
        2 * criterion(
            x_hat[:n_sub_sp],
            sub_exp_sp
        )
        +
        4 * criterion(
            x_hat[n_sub_sp:n_sub_sp + n_sub_sc],
            sub_exp_sc
        )
    )

    total_loss = (
        distribution_loss +
        reconstruction_loss
    )

    return (
        total_loss,
        distribution_loss,
        reconstruction_loss
    )



def train_subgraph_model(
        model,
        optimizer,
        grid_subgraphs,
        features_cpu,
        targets_cpu,
        device,
        criterion,
        epochs,
        m,
        weight=5,
        random_generator=None,
        print_every=100,
):

    if len(grid_subgraphs) == 0:
        raise ValueError(
            "grid_subgraphs为空，无法进行训练。"
        )

    if m <= 0:
        raise ValueError(
            f"m必须大于0，当前m={m}。"
        )

    if m > len(grid_subgraphs):
        raise ValueError(
            f"当前只有{len(grid_subgraphs)}个有效子图，"
            f"但每轮要求抽取m={m}个子图。"
        )

    if random_generator is None:
        random_generator = np.random.default_rng()

    model.train()

    for epoch_index in range(epochs):

        optimizer.zero_grad(
            set_to_none=True
        )

        selected_subgraph_indices = (
            random_generator.choice(
                len(grid_subgraphs),
                size=m,
                replace=False
            )
        )

        epoch_total_loss = 0.0
        epoch_distribution_loss = 0.0
        epoch_reconstruction_loss = 0.0

        for subgraph_index in selected_subgraph_indices:

            subgraph = grid_subgraphs[
                int(subgraph_index)
            ]

            (
                total_loss,
                distribution_loss,
                reconstruction_loss
            ) = calculate_subgraph_loss(
                model=model,
                subgraph=subgraph,
                features_cpu=features_cpu,
                targets_cpu=targets_cpu,
                device=device,
                criterion=criterion,
                weight=weight
            )

            (total_loss / m).backward()

            epoch_total_loss += (
                total_loss.detach().item()
            )

            epoch_distribution_loss += (
                distribution_loss.detach().item()
            )

            epoch_reconstruction_loss += (
                reconstruction_loss.detach().item()
            )

        optimizer.step()

        if (
            print_every is not None
            and print_every > 0
            and epoch_index % print_every == 0
        ):
            print(
                f"Epoch {epoch_index}, "
                f"Loss: {epoch_total_loss / m:.4f}, "
                f"Distribution: "
                f"{epoch_distribution_loss / m:.4f}, "
                f"Reconstruction: "
                f"{epoch_reconstruction_loss / m:.4f}"
            )

    return model



def micro_network_training_subgraph(
        adata_sp,
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
        num_layers=2,
        weight=5,
        k=10,
        m=4,
        spatial_key='spatial',
        random_seed=0,
):


    if device == 'auto':
        device = torch.device(
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )
    else:
        device = torch.device(device)

    n_sp = adata_sp.n_obs
    n_sc = adata_sc.n_obs
    n_nodes = n_sp + n_sc


    if hasattr(features, "toarray"):
        features_np = features.toarray()
    else:
        features_np = np.asarray(features)

    if hasattr(targets, "toarray"):
        targets_np = targets.toarray()
    else:
        targets_np = np.asarray(targets)


    features_cpu = torch.as_tensor(
        features_np,
        dtype=torch.float32
    )

    targets_cpu = torch.as_tensor(
        targets_np,
        dtype=torch.float32
    )


    full_edge_index = torch.as_tensor(
        edge_index,
        dtype=torch.long
    )

    full_edge_index = full_edge_index.t().contiguous()

    full_edge_index = to_undirected(
        full_edge_index,
        num_nodes=n_nodes
    )

    full_edge_index, _ = add_self_loops(
        full_edge_index,
        num_nodes=n_nodes
    )

    full_edge_index = (
        full_edge_index
        .cpu()
        .long()
        .contiguous()
    )


    grid_subgraphs = build_grid_subgraphs(
        adata_sp=adata_sp,
        adata_sc=adata_sc,
        full_edge_index=full_edge_index,
        k=k,
        spatial_key=spatial_key
    )

    if m > len(grid_subgraphs):
        raise ValueError(
            f"当前只有{len(grid_subgraphs)}个有效网格子图，"
            f"但m={m}。"
        )

    print(
        f"网格划分完成：{k} × {k}，"
        f"有效子图数量：{len(grid_subgraphs)}，"
        f"每轮随机抽取：{m}个子图。"
    )

    input_features_num_1 = features_np.shape[1]
    output_features_num_1 = targets_np.shape[1]

    input_features_num_2 = targets_np.shape[1]
    output_features_num_2 = features_np.shape[1]

    encoder = GATEncoder(
        input_features_num_1,
        output_features_num_1,
        hidden=hidden_dim,
        heads=heads,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)

    decoder = GATDecoder(
        input_features_num_2,
        output_features_num_2,
        hidden=hidden_dim,
        heads=heads,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)

    model = GraphAutoencoder(
        encoder,
        decoder
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.MSELoss()

    random_generator = np.random.default_rng(
        random_seed
    )

    model = train_subgraph_model(
        model=model,
        optimizer=optimizer,
        grid_subgraphs=grid_subgraphs,
        features_cpu=features_cpu,
        targets_cpu=targets_cpu,
        device=device,
        criterion=criterion,
        epochs=epoch,
        m=m,
        weight=weight,
        random_generator=random_generator,
        print_every=100
    )

    model.eval()

    with torch.no_grad():

        full_features_device = features_cpu.to(
            device,
            non_blocking=True
        )

        full_edge_index_device = full_edge_index.to(
            device,
            non_blocking=True
        )

        _, micro_embeddings = model(
            full_features_device,
            full_edge_index_device
        )

        micro_embeddings = (
            micro_embeddings
            .cpu()
            .numpy()
        )


    adata_sc.obsm['embedding'] = np.hstack([
        features_np[n_sp:],
        micro_embeddings[n_sp:]
    ])

    adata_sp.obsm['embedding'] = np.hstack([
        features_np[:n_sp],
        micro_embeddings[:n_sp]
    ])

    return model