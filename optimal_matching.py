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
    correlation_matrix = np.zeros((n_rows, n_cols))
    def _compute_correlation(index1, vec1, banksy_train):
        return [pearsonr(vec1, vec2)[0] for vec2 in banksy_train]

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(delayed(_compute_correlation)(index1, vec1, micro_embedding_sp)
                                  for index1, vec1 in tqdm(enumerate(micro_embedding_sc), total=n_rows))
    for index1, result in enumerate(results):
        correlation_matrix[index1, :] = result

    cell_type_counts = adata_sc.obs['cell_type'].value_counts()
    cell_type_proportions = cell_type_counts / adata_sc.shape[0]
    sampling_counts = cell_type_counts

    adata_sc.obsm['pseudo_space_cor'] = np.zeros((len(adata_sc), 2))

    for iteration in range(repetition):
        sampled_indices = []
        st_counts = adata_sp.obs['cell_type'].value_counts()
        sc_counts = adata_sp.obs['cell_type'].value_counts()

        all_sufficient = all(
            (cell_type in sc_counts) and (sc_counts[cell_type] >= st_counts.get(cell_type, 0))
            for cell_type in st_counts.index
        )

        for cell_type, count in sampling_counts.items():
            cell_type_names = adata_sp.obs[adata_sp.obs['cell_type'] == cell_type].index
            cell_indices = [np.where(adata_sp.obs.index == name)[0][0] for name in cell_type_names]
            if all_sufficient:
                sampled = list(cell_indices)
                additional_samples = np.random.choice(cell_indices, size=count - len(cell_indices), replace=True)
                sampled.extend(additional_samples)
            else:
                sampled = np.random.choice(cell_indices, size=count, replace=True)
            sampled_indices.extend(sampled)
        sampled_indices = np.array(sampled_indices)

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
    correlation_matrix = np.zeros((n_rows, n_cols))
    def _compute_correlation(index1, vec1, banksy_train):
        return [pearsonr(vec1, vec2)[0] for vec2 in banksy_train]

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(delayed(_compute_correlation)(index1, vec1, micro_embedding_sp)
                                  for index1, vec1 in tqdm(enumerate(micro_embedding_sc), total=n_rows))
    for index1, result in enumerate(results):
        correlation_matrix[index1, :] = result

    cell_type_counts = adata_sc.obs['cell_type'].value_counts()
    cell_type_proportions = cell_type_counts / adata_sc.shape[0]
    sampling_counts = cell_type_counts

    adata_sc.obsm['pseudo_space'] = np.zeros((len(adata_sc), 2))

    sampled_indices = []
    st_counts = adata_sp.obs['cell_type'].value_counts()
    sc_counts = adata_sp.obs['cell_type'].value_counts()

    all_sufficient = all(
        (cell_type in sc_counts) and (sc_counts[cell_type] >= st_counts.get(cell_type, 0))
        for cell_type in st_counts.index
    )

    for cell_type, count in sampling_counts.items():
        cell_type_names = adata_sp.obs[adata_sp.obs['cell_type'] == cell_type].index
        cell_indices = [np.where(adata_sp.obs.index == name)[0][0] for name in cell_type_names]
        if all_sufficient:
            sampled = list(cell_indices)
            additional_samples = np.random.choice(cell_indices, size=count - len(cell_indices), replace=True)
            sampled.extend(additional_samples)
        else:
            sampled = np.random.choice(cell_indices, size=count, replace=True)
        sampled_indices.extend(sampled)
    sampled_indices = np.array(sampled_indices)

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



def matching(adata_sp,
                    adata_sc,
                    spatial_key='spatial',
                    mode='perturbation',
                    repetition=100,
                    random_state=42,
                    scale=20):

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
        raise ValueError(" Mode needs to be 'perturbation' or 'repetition'. ")
