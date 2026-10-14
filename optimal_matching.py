from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
import scanpy as sc


def matching_repetition(adata_sp,
                    adata_sc,
                    spatial_key='spatial',
                    repetition=100):
    n_rows = len(adata_sc.obs_names)
    n_cols = len(adata_sp.obs_names)
    micro_embedding_sc = adata_sc.obsm['embedding']
    micro_embedding_sp = adata_sp.obsm['embedding']

    X = np.asarray(micro_embedding_sc, dtype=np.float32)
    Y = np.asarray(micro_embedding_sp, dtype=np.float32)

    X_centered = X - X.mean(axis=1, keepdims=True)
    Y_centered = Y - Y.mean(axis=1, keepdims=True)

    X_norm = np.linalg.norm(X_centered, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_centered, axis=1, keepdims=True)

    X_normalized = X_centered / X_norm
    Y_normalized = Y_centered / Y_norm

    correlation_matrix = X_normalized @ Y_normalized.T

    cell_type_counts = adata_sc.obs['cell_type'].value_counts()
    cell_type_proportions = cell_type_counts / adata_sc.shape[0]
    sampling_counts = cell_type_counts

    adata_sc.obsm['pseudo_space_cor'] = np.zeros((len(adata_sc), 2))

    for iteration in range(repetition):
    
        sampled_indices = []
        
        for cell_type, count in sampling_counts.items():
            count = int(count)

            cell_indices = np.flatnonzero(
                adata_sp.obs['cell_type'].to_numpy() == cell_type
            )

            n_available = len(cell_indices)

            if n_available == 0:
                raise ValueError(
                    f"Cell type '{cell_type}' exists in adata_sc "
                    f"but is absent from adata_sp."
                )

            if n_available >= count:

                sampled = np.random.choice(
                    cell_indices,
                    size=count,
                    replace=False
                )
            else:

                additional_samples = np.random.choice(
                    cell_indices,
                    size=count - n_available,
                    replace=True
                )

                sampled = np.concatenate([
                    cell_indices,
                    additional_samples
                ])

            sampled_indices.extend(sampled.tolist())

        sampled_indices = np.asarray(sampled_indices, dtype=int)

        columns = []
        for idx in sampled_indices:
            columns.append(correlation_matrix[:, idx])

        correlation_matrix_imputation = np.column_stack(columns)
        correlation_matrix_imputation = correlation_matrix_imputation + 1

        def _maximum_weight_matching(cost_matrix):
            row_indices, col_indices = linear_sum_assignment(-cost_matrix)
            matching_pairs = np.array(list(zip(row_indices, col_indices)))

            return matching_pairs

        matching_pairs = _maximum_weight_matching(correlation_matrix_imputation)

        for i in range(n_rows):
            adata_sc.obsm['pseudo_space_cor'][i] += adata_sp.obsm[spatial_key][sampled_indices[matching_pairs[i, 1]]]

    adata_sc.obsm['pseudo_space'] = adata_sc.obsm['pseudo_space_cor'] / repetition

    return adata_sc


def matching_perturbation(adata_sp,
                    adata_sc,
                    spatial_key='spatial',
                    random_state=42,
                    scale=20):
    np.random.seed(random_state)
    n_rows = len(adata_sc.obs_names)
    n_cols = len(adata_sp.obs_names)
    micro_embedding_sc = adata_sc.obsm['embedding']
    micro_embedding_sp = adata_sp.obsm['embedding']

    X = np.asarray(micro_embedding_sc, dtype=np.float32)
    Y = np.asarray(micro_embedding_sp, dtype=np.float32)

    X_centered = X - X.mean(axis=1, keepdims=True)
    Y_centered = Y - Y.mean(axis=1, keepdims=True)

    X_norm = np.linalg.norm(X_centered, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_centered, axis=1, keepdims=True)

    X_normalized = X_centered / X_norm
    Y_normalized = Y_centered / Y_norm

    correlation_matrix = X_normalized @ Y_normalized.T

    cell_type_counts = adata_sc.obs['cell_type'].value_counts()
    cell_type_proportions = cell_type_counts / adata_sc.shape[0]

    sampling_counts = cell_type_counts

    adata_sc.obsm['pseudo_space'] = np.zeros((len(adata_sc), 2))

    sampled_indices = []

    np.random.seed(random_state)

    for cell_type, count in sampling_counts.items():
        count = int(count)

        cell_indices = np.flatnonzero(
            adata_sp.obs['cell_type'].to_numpy() == cell_type
        )

        n_available = len(cell_indices)

        if n_available == 0:
            raise ValueError(
                f"Cell type '{cell_type}' exists in adata_sc "
                f"but is absent from adata_sp."
            )

        if n_available >= count:

            sampled = np.random.choice(
                cell_indices,
                size=count,
                replace=False
            )
        else:

            additional_samples = np.random.choice(
                cell_indices,
                size=count - n_available,
                replace=True
            )

            sampled = np.concatenate([
                cell_indices,
                additional_samples
            ])

        sampled_indices.extend(sampled.tolist())

    sampled_indices = np.asarray(sampled_indices, dtype=int)

    columns = []
    for idx in sampled_indices:
        columns.append(correlation_matrix[:, idx])

    correlation_matrix_imputation = np.column_stack(columns)
    correlation_matrix_imputation = correlation_matrix_imputation + 1

    def _maximum_weight_matching(cost_matrix):
        row_indices, col_indices = linear_sum_assignment(-cost_matrix)
        matching_pairs = np.array(list(zip(row_indices, col_indices)))

        return matching_pairs

    matching_pairs = _maximum_weight_matching(correlation_matrix_imputation)

    for i in range(n_rows):
        adata_sc.obsm['pseudo_space'][i] += adata_sp.obsm[spatial_key][sampled_indices[matching_pairs[i, 1]]]

    adata_sc.obsm['pseudo_space'] += np.random.normal(0, scale=scale, size=adata_sc.obsm['pseudo_space'].shape)

    return adata_sc



def matching_sample_sc(
    adata_sp,
    adata_sc,
    spatial_key='spatial',
    random_state=42,
    scale=20,
    k=2
):

    np.random.seed(random_state)

    sp_celltype_counts = (
        adata_sp.obs['cell_type']
        .value_counts()
    )

    sampled_sc_indices = []

    for cell_type, sp_count in sp_celltype_counts.items():

        target_count = int(sp_count * k)

        sc_indices = np.flatnonzero(
            adata_sc.obs['cell_type'].to_numpy()
            == cell_type
        )

        n_available = len(sc_indices)

        if n_available == 0:
            raise ValueError(
                f"Cell type '{cell_type}' exists in adata_sp "
                f"but is absent from adata_sc."
            )

        if n_available >= target_count:

            sampled = np.random.choice(
                sc_indices,
                size=target_count,
                replace=False
            )

        else:

            additional_samples = np.random.choice(
                sc_indices,
                size=target_count - n_available,
                replace=True
            )

            sampled = np.concatenate([
                sc_indices,
                additional_samples
            ])

        sampled_sc_indices.extend(
            sampled.tolist()
        )

    sampled_sc_indices = np.asarray(sampled_sc_indices, dtype=int)

    adata_sc_sampled = adata_sc[sampled_sc_indices].copy()
    adata_sc_sampled.obs_names_make_unique()

    expanded_sp_indices = np.repeat(np.arange(adata_sp.n_obs),k)


    micro_embedding_sc = adata_sc_sampled.obsm['embedding']
    micro_embedding_sp = adata_sp.obsm['embedding']

    X = np.asarray(micro_embedding_sc, dtype=np.float32)
    Y = np.asarray(micro_embedding_sp, dtype=np.float32)

    X_centered = X- X.mean(axis=1, keepdims=True)
    Y_centered = Y- Y.mean(axis=1, keepdims=True)

    X_norm = np.linalg.norm(X_centered, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_centered, axis=1, keepdims=True)

    X_normalized = X_centered / X_norm
    Y_normalized = Y_centered / Y_norm

    correlation_matrix = X_normalized @ Y_normalized.T

    correlation_matrix_expanded = (
        correlation_matrix[
            :,
            expanded_sp_indices
        ]
    )

    correlation_matrix_expanded = (
        correlation_matrix_expanded
        + 1
    )

    print("sampled adata_sc:", adata_sc_sampled.n_obs)
    print("expanded adata_sp:", len(expanded_sp_indices))
    print("correlation matrix:", correlation_matrix_expanded.shape)

    def _maximum_weight_matching(cost_matrix):
        row_indices, col_indices = linear_sum_assignment(-cost_matrix)
        matching_pairs = np.column_stack([row_indices,col_indices])

        return matching_pairs


    matching_pairs = (
        _maximum_weight_matching(
            correlation_matrix_expanded
        )
    )


    matched_expanded_sp_idx = matching_pairs[:, 1]
    matched_original_sp_idx = expanded_sp_indices[matched_expanded_sp_idx]

    adata_sc_sampled.obsm['pseudo_space'] = np.zeros((adata_sc_sampled.n_obs, 2), dtype=float)
    adata_sc_sampled.obsm['pseudo_space'][matching_pairs[:, 0]] = adata_sp.obsm[spatial_key][matched_original_sp_idx]
    adata_sc_sampled.obsm['pseudo_space'] += np.random.normal(0, scale=scale, size=adata_sc_sampled.obsm['pseudo_space'].shape)

    return adata_sc_sampled



from typing import Callable, Optional


def matching(
    adata_sp,
    adata_sc,
    spatial_key='spatial',
    mode='perturbation',
    repetition=100,
    random_state=42,
    scale=20,
    sample='st',
    k=2,
    matching_func: Optional[Callable] = None
):

    if matching_func is not None:

        if not callable(matching_func):
            raise TypeError(
                "matching_func must be callable."
            )

        print(" Use custom matching function. ")

        return matching_func(
            adata_sp=adata_sp,
            adata_sc=adata_sc,
            spatial_key=spatial_key
        )

    if sample == 'sc':

        print(" Use 'sc sampling' mode. ")

        return matching_sample_sc(
            adata_sp=adata_sp,
            adata_sc=adata_sc,
            spatial_key=spatial_key,
            random_state=random_state,
            scale=scale,
            k=k
        )

    elif sample == 'st':

        if mode == 'perturbation':

            print(" Use 'perturbation' mode. ")

            return matching_perturbation(
                adata_sp=adata_sp,
                adata_sc=adata_sc,
                spatial_key=spatial_key,
                random_state=random_state,
                scale=scale
            )

        elif mode == 'repetition':

            print(" Use 'repetition' mode. ")

            return matching_repetition(
                adata_sp=adata_sp,
                adata_sc=adata_sc,
                spatial_key=spatial_key,
                repetition=repetition
            )

        else:

            raise ValueError(
                " Mode needs to be 'perturbation' or 'repetition'. "
            )

    else:

        raise ValueError(
            " sample needs to be 'st' or 'sc'. "
        )
