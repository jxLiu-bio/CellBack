## Installation
### 1. Create a virtual environment (recommended)
We recommend using conda to manage dependencies by:

```bash
conda create -n cellback_env python=3.9
conda activate cellback_env
```
### 2. Install CellBack
Download the CellBack package from GitHub:
```bash
git clone https://github.com/jxLiu-bio/CellBack.git
cd CellBack
```
and install by:
```bash
pip install .
```

### 3. Install PyTorch and PyTorch Geometric
Install PyTorch according to your hardware configuration. For example, you can run:

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121   --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
```
More information can be found at: [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


### 4. Install Seurat
Seurat is used to reconstruct a shared expression matrix between scRNA-seq and ST data,
which is subsequently used to derive the microenvironment matrix. You can install Seurat in the virtual environment by conda:
```bash
conda install -c bioconda r-seurat
```

### 5. Install scvi-tools
scANVI is employed to learn batch-corrected low-dimensional embeddings of cells,
which are subsequently used to construct the cell–cell adjacency graph for input into the graph attention network. Install scvi-tools in the virtual environment by pip:
```bash
pip install scvi-tools==1.15.0
