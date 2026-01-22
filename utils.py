import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import warnings
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
import scipy.sparse as ss


def run_banksy(adata,
               spatial_key='spatial',
               k_geom=15,
               max_m=1,
               nbr_weight_decay="scaled_gaussian",
               lambda_list=None):
    if lambda_list is None:
        raise ValueError("lambda_list must be provided from run_banksy().")

    from .banksy.embed_banksy import generate_banksy_matrix
    from .banksy.banksy_utils.plot_utils import plot_cell_positions
    from .banksy.main import median_dist_to_nearest_neighbour

    adata_sp = adata.copy()
    coord_keys = ('xcoord', 'ycoord', 'coord_xy')
    adata_sp.obs['xcoord'] = np.array(adata_sp.obsm[spatial_key][:, 0])
    adata_sp.obs['ycoord'] = np.array(adata_sp.obsm[spatial_key][:, 1])
    raw_x = adata_sp.obs['xcoord']
    raw_y = adata_sp.obs['ycoord']
    adata_sp.obsm['coord_xy'] = adata_sp.obsm[spatial_key]

    nbrs = median_dist_to_nearest_neighbour(adata_sp, key=coord_keys[2])

    from .banksy.initialize_banksy import initialize_banksy
    banksy_dict = initialize_banksy(
        adata_sp,
        coord_keys,
        k_geom,
        nbr_weight_decay=nbr_weight_decay,
        max_m=max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    plt.show()

    banksy_dict, adata_sp = generate_banksy_matrix(adata_sp,
                                                   banksy_dict,
                                                   lambda_list,
                                                   max_m)

    adata_sp.obsm[spatial_key] = adata.obsm[spatial_key]

    return adata_sp, lambda_list[0]


def inter_correlation_calculation(adata_sc,
                                  adata_sp,
                                  basis='X_pca',
                                  k=5):
    from joblib import Parallel, delayed
    correlation_results = np.zeros((len(adata_sc.obs_names),len(adata_sp.obs_names)))
    def _compute_correlation(index1, vec1, data):
        return [pearsonr(vec1, vec2)[0] for vec2 in data]

    results = Parallel(n_jobs=-1)(delayed(_compute_correlation)(index1, vec1, adata_sp.obsm[basis]) for index1,
    vec1 in tqdm(enumerate(adata_sc.obsm[basis]), total=len(adata_sc.obs_names)))
    for index1, result in enumerate(results):
        correlation_results[index1, :] = result

    edge_index = []
    for i in range(len(adata_sc.obs_names)):
        top_indices = np.argsort(correlation_results[i, :])[-k:][::-1]
        for l in range(k):
            edge_index.append([top_indices[l], i + adata_sp.shape[0]])
            edge_index.append([i + adata_sp.shape[0], top_indices[l]])

    return edge_index


def intra_spatial_proximity(adata_sp,
                            spatial_key='spatial',
                            k=5,
                            threshold_distance=None,
                            ):
    distance_matrix = np.zeros((len(adata_sp.obs_names), len(adata_sp.obs_names)))
    for index1, vec1 in tqdm(enumerate(adata_sp.obsm[spatial_key]),
                             total=adata_sp.obsm[spatial_key].shape[0]):
        for index2, vec2 in enumerate(adata_sp.obsm[spatial_key]):
            distance = np.linalg.norm(vec1 - vec2)
            distance_matrix[index1, index2] = distance

    edge_index = []
    if threshold_distance is not None:
        flat_matrix_distance = distance_matrix.flatten()
        count_distance = np.sum(flat_matrix_distance < threshold_distance) / adata_sp.shape[0] / 2
        print(f"Mean degree(neighbors): {count_distance}")

        for i in range(adata_sp.shape[0]):
            for j in range(adata_sp.shape[0]):
                if i != j and distance_matrix[i, j] < threshold_distance:
                    edge_index.append([i, j])
    else:
        for i in range(len(adata_sp.obs_names)):
            top_indices = np.argsort(distance_matrix[i, :], )[:k][::-1]
            for l in range(k):
                edge_index.append([top_indices[l], k])
                edge_index.append([k, top_indices[l]])

    return edge_index


def extra_features(adata_sc,
                   adata_sp,
                   adata_sp_micro,
                   lam=None,
                   ):
    if lam is None:
        raise ValueError("lam must equal micro_weight.")

    features_sp = adata_sp.X
    if ss.issparse(features_sp):
        features_sp = features_sp.toarray()
    features_sc = adata_sc.X
    if ss.issparse(features_sc):
        features_sc = features_sc.toarray()

    def _standardize_columns(matrix):
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        standardized_matrix = (matrix - mean) / std
        standardized_matrix = np.nan_to_num(standardized_matrix)
        return standardized_matrix
    features_sc = _standardize_columns(features_sc) * np.sqrt(1 - lam)
    features_sp = _standardize_columns(features_sp) * np.sqrt(1 - lam)

    features = np.vstack((features_sp, features_sc))
    targets = adata_sp_micro[:, int(adata_sp_micro.shape[1] / 3):].X
    if ss.issparse(targets):
        targets = targets.toarray()

    return features, targets


def run_scvi(slide,
             singlecell,
             run_hvg: bool = True,
             ):

    import scvi
    import anndata as ad

    batch_categories = ["slide", "singlecell"]
    adata_list = [slide, singlecell]
    merged_data = ad.concat(adata_list, index_unique=None, label="batch", keys=batch_categories)
    adata = merged_data
    adata.layers["counts"] = adata.X
    adata.raw = adata
    if run_hvg:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=2000,
            layer="counts",
            batch_key="batch",
            subset=True,
        )
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=15, n_neighbors=30)
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(adata, color=["batch", "cell_type"])

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=50, gene_likelihood="nb")
    model.train()

    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_scVI")
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(adata, color=["batch", "cell_type"])

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key="cell_type",
        unlabeled_category="Unknown",
    )
    scanvi_model.train(max_epochs=20, n_samples_per_label=100)
    SCANVI_LATENT_KEY = "X_scANVI"
    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_scANVI")
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(adata, color=["batch", "cell_type"])

    adata_sp = adata[adata.obs['batch'] == "slide"].copy()
    adata_sc = adata[adata.obs['batch'] == "singlecell"].copy()
    adata_sp.obsm['spatial'] = slide.obsm['coordinate']

    return adata_sc, adata_sp

def run_seurat(slide,
               singlecell,
               dictory_sc= "file_sc",
               dictory_st= "file_st",
               dictory_matrix= "file_matrix",
               ):

    import os
    from scipy import sparse

    os.makedirs(dictory_st, exist_ok=True)
    os.makedirs(dictory_sc, exist_ok=True)
    os.makedirs(dictory_matrix, exist_ok=True)

    X = slide.X.toarray() if sparse.issparse(slide.X) else slide.X
    df = pd.DataFrame(X)
    df.to_csv(f'{dictory_st}/expression.csv', index=False)
    df = pd.DataFrame(slide.obs_names)
    df.to_csv(f'{dictory_st}/obs_names.csv', index=False)
    df = pd.DataFrame(slide.var_names)
    df.to_csv(f'{dictory_st}/var_names.csv', index=False)

    X = singlecell.X.toarray() if sparse.issparse(singlecell.X) else singlecell.X
    df = pd.DataFrame(X)
    df.to_csv(f'{dictory_sc}/expression.csv', index=False)
    df = pd.DataFrame(singlecell.obs_names)
    df.to_csv(f'{dictory_sc}/obs_names.csv', index=False)
    df = pd.DataFrame(singlecell.var_names)
    df.to_csv(f'{dictory_sc}/var_names.csv', index=False)

    from rpy2.robjects.packages import importr

    Seurat = importr('Seurat')
    from rpy2 import robjects
    robjects.r('library(Seurat)')
    version = robjects.r('packageVersion("Seurat")')[0]
    print(version)

    from rpy2.robjects import r, globalenv

    globalenv["dictory_st"] = dictory_st
    globalenv["dictory_sc"] = dictory_sc
    globalenv["dictory_matrix"] = dictory_matrix

    r("""
    library(Seurat)
    library(ggplot2)

    expression_matrix <- read.csv(file.path(dictory_st, "expression.csv"))
    cell_names <- read.csv(file.path(dictory_st, "obs_names.csv"))
    gene_names <- read.csv(file.path(dictory_st, "var_names.csv"))
    rownames(expression_matrix) <- cell_names[, 1]
    colnames(expression_matrix) <- gene_names[, 1]
    ref <- CreateSeuratObject(counts = t(expression_matrix))

    expression_matrix <- read.csv(file.path(dictory_sc, "expression.csv"))
    cell_names <- read.csv(file.path(dictory_sc, "obs_names.csv"))
    gene_names <- read.csv(file.path(dictory_sc, "var_names.csv"))
    rownames(expression_matrix) <- cell_names[, 1]
    colnames(expression_matrix) <- gene_names[, 1]
    seekspace <- CreateSeuratObject(counts = t(expression_matrix))

    seekspace$tech <- "seekspace"
    ref$tech <- "ref"

    combined_seurat_obj <- merge(seekspace, y = ref, add.cell.ids = c("seekspace", "ref"))
    combined_seurat_obj.list <- SplitObject(combined_seurat_obj, split.by = "tech")

    combined_seurat_obj <- ScaleData(combined_seurat_obj)
    combined_seurat_obj <- RunPCA(combined_seurat_obj, features = rownames(combined_seurat_obj))
    combined_seurat_obj <- FindNeighbors(combined_seurat_obj, dims = 1:50)
    combined_seurat_obj <- FindClusters(combined_seurat_obj, resolution = 0.5)
    combined_seurat_obj <- RunUMAP(combined_seurat_obj, dims = 1:50)
    DimPlot(combined_seurat_obj, group.by = c("tech"))

    features <- SelectIntegrationFeatures(object.list = combined_seurat_obj.list)
    combined_seurat_obj.list <- lapply(X = combined_seurat_obj.list, FUN = function(x) {
      x <- ScaleData(x, features = features, verbose = FALSE)
      x <- RunPCA(x, features = features, verbose = FALSE)
    })

    anchors <- FindIntegrationAnchors(object.list = combined_seurat_obj.list, reference = 2, reduction = "rpca",
                                      dims = 1:50)
    combined_seurat_obj.integrated <- IntegrateData(anchorset = anchors, dims = 1:50)

    combined_seurat_obj.integrated <- ScaleData(combined_seurat_obj.integrated)
    combined_seurat_obj.integrated <- RunPCA(combined_seurat_obj.integrated)
    combined_seurat_obj.integrated <- FindNeighbors(combined_seurat_obj.integrated, dims = 1:50)
    combined_seurat_obj.integrated <- FindClusters(combined_seurat_obj.integrated, resolution = 0.5)
    combined_seurat_obj.integrated <- RunUMAP(combined_seurat_obj.integrated, dims = 1:50)
    DimPlot(combined_seurat_obj.integrated, group.by = c("tech"))

    dense_mat <- as.matrix(combined_seurat_obj.integrated@assays$integrated@data)
    write.csv(dense_mat, file = file.path(dictory_matrix, "reconstruction_expression.csv"), row.names = FALSE)
    """)

    from anndata import AnnData

    expression = pd.read_csv(
        f'{dictory_matrix}/reconstruction_expression.csv')
    expression = expression.to_numpy().T

    adata_sp_re = AnnData(expression[len(singlecell.obs_names):, :])
    adata_sp_re.obs_names = slide.obs_names
    adata_sp_re.var_names = slide.var_names
    adata_sp_re.obsm['spatial'] = slide.obsm['spatial']
    adata_sp_re.obs['cell_type'] = slide.obs['cell_type']
    adata_sp_re.obsm['X_scANVI'] = slide.obsm['X_scANVI']

    adata_sc_re = AnnData(expression[:len(singlecell.obs_names), :])
    adata_sc_re.obs_names = singlecell.obs_names
    adata_sc_re.var_names = singlecell.var_names
    adata_sc_re.obs['cell_type'] = singlecell.obs['cell_type']
    adata_sc_re.obsm['X_scANVI'] = singlecell.obsm['X_scANVI']

    return adata_sc_re, adata_sp_re


def denoising(adata_sp,
              adata_sc,
              singlecell_X,
              slide_ori,
              k=100):
    from scipy import sparse

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

    n = k
    indices_spatial = np.argsort(correlation_matrix, axis=0)[-n:][::-1]
    indices_spatial = indices_spatial.T

    cell_types_singlecell = adata_sc.obs['cell_type'].to_numpy()
    cell_types_slide = adata_sp.obs['cell_type'].to_numpy()

    average_expression = np.zeros((len(slide_ori.obs_names), len(slide_ori.var_names)))
    slide_X = slide_ori.X.toarray() if sparse.issparse(slide_ori.X) else slide_ori.X

    for i, neighbor_indices in tqdm(enumerate(indices_spatial), total=len(slide_ori.obs_names)):
        neighbor_expression = singlecell_X[neighbor_indices, :]
        mask = cell_types_singlecell[neighbor_indices] == cell_types_slide[i]

        if np.sum(mask) == 0:
            average_expression[i, :] = slide_X[i, :]
            continue

        neighbor_expression = neighbor_expression[mask, :]
        weights = correlation_matrix[neighbor_indices, i][mask]
        normalized_weights = weights / np.sum(weights)
        average_expression[i, :] = np.dot(normalized_weights, neighbor_expression)

    import anndata as ad

    adata_denoise = ad.AnnData(average_expression)
    adata_denoise.obs_names = slide_ori.obs_names
    adata_denoise.var_names = slide_ori.var_names
    adata_denoise.obsm['spatial'] = slide_ori.obsm['coordinate']
    adata_denoise.obs['cell_type'] = slide_ori.obs['cell_type']

    return adata_denoise

def imputation_micro(adata_sp,
              adata_sc,
              slide,
              singlecell,
              train_list,
              k=1000,
              num=30000,
              device="auto",
              random_state=66,
              ):

    import torch
    import random
    seed = random_state
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    n = k
    indices_spatial = np.argsort(correlation_matrix, axis=0)[-n:][::-1]
    indices_spatial = indices_spatial.T

    from torch.nn.functional import softmax, cosine_similarity

    matrix = np.random.normal(0, 1, (slide.shape[0], indices_spatial.shape[1]))
    matrix = torch.tensor(matrix, device=device, requires_grad=True, dtype=torch.float32)
    slide_ = slide[:, train_list]
    singlecell_ = singlecell[:, train_list]

    optimizer = torch.optim.Adam([matrix], lr=0.1)
    S = torch.tensor(singlecell_.X, device=device, dtype=torch.float32)
    G = torch.tensor(slide_.X, device=device, dtype=torch.float32)
    indices_spatial = np.array(indices_spatial)
    indices_spatial = torch.tensor(indices_spatial, dtype=torch.long)

    for epoch in range(num):
        optimizer.zero_grad()
        M_probs = softmax(matrix, dim=1)
        S_selected = S[indices_spatial]
        M_expanded = M_probs.unsqueeze(1)
        G_pred = torch.bmm(M_expanded, S_selected).squeeze(1)
        gv_term = cosine_similarity(G_pred, G, dim=0).mean()
        regularizer_term = -((torch.log(M_probs) * M_probs).sum()) / slide.shape[0]
        loss = - gv_term - 1e-2 * regularizer_term
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    with torch.no_grad():
        output = softmax(matrix, dim=1).cpu().numpy()

    G_predicted = np.zeros((slide.shape[0], singlecell.X.shape[1]))
    for i in range(slide.shape[0]):
        selected_indices = indices_spatial[i]
        S_selected = singlecell.X[selected_indices]
        G_predicted[i] = output[i] @ S_selected

    return G_predicted


def imputation_no_micro(slide,
              singlecell,
              train_list,
              num=1000,
              device="auto",
              random_state = 66,
              ):

    import torch
    import random
    seed = random_state
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torch.nn.functional import softmax, cosine_similarity

    slide_ = slide[:, train_list]
    singlecell_ = singlecell[:, train_list]
    matrix = np.random.normal(0, 1, (singlecell_.shape[0], slide_.shape[0]))
    matrix = torch.tensor(matrix, device=device, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam([matrix], lr=0.1)
    S = torch.tensor(singlecell_.X, device=device, dtype=torch.float32)
    G = torch.tensor(slide_.X, device=device, dtype=torch.float32)

    for epoch in range(num):
        optimizer.zero_grad()
        M_probs = softmax(matrix, dim=1)
        G_pred = torch.matmul(M_probs.t(), S)
        gv_term = cosine_similarity(G_pred, G, dim=0).mean()
        regularizer_term = -((torch.log(M_probs) * M_probs).sum()) / slide.shape[0]
        loss = - 1 * gv_term - 1e-2 * regularizer_term
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(
                f'Epoch {epoch}, Loss: {loss.item():.4f}')

    with torch.no_grad():
        output = softmax(matrix, dim=1).cpu().numpy()

    G_predicted = output.T @ singlecell.X

    return G_predicted

def gene_imputation(
    mode,
    adata_sp=None,
    adata_sc=None,
    slide=None,
    singlecell=None,
    train_list=None,
    k=1000,
    num=30000,
    device="auto",
    random_state=66
):

    if mode == "micro":
        if adata_sp is None or adata_sc is None:
            raise ValueError("adata_sp and adata_sc must be provided for mode='micro'")

        return imputation_micro(
            adata_sp=adata_sp,
            adata_sc=adata_sc,
            slide=slide,
            singlecell=singlecell,
            train_list=train_list,
            k=k,
            num=num,
            device=device,
            random_state=random_state,
        )

    elif mode == "no_micro":
        return imputation_no_micro(
            slide=slide,
            singlecell=singlecell,
            train_list=train_list,
            num=num,
            device=device,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            "Available modes are ['micro', 'no_micro']"
        )

def cell_imputation(adata_denoised,
                    singlecell
                    ):
    adata_imputation = singlecell.concatenate(adata_denoised, join='inner', batch_key='batch')
    return adata_imputation