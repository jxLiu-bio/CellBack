## About

**CellBack** is a Python package for **microenvironment-aware cell imputation** in single-cell whole-transcriptome spatial transcriptomics (scWT-ST) data.

CellBack reconstructs complete tissue maps at **single-cell resolution** by integrating **matched scRNA-seq data** with **spatial information** and **microenvironmental context**.

### Workflow

1. **Microenvironment Representation Learning**
   - Reconstructs expression matrices of shared highly variable genes between ST and scRNA-seq data (optional batch correction using Seurat).  
   - Generates **microenvironment representations** for ST cells using **BANKSY**.  
   - Constructs a cross-platform graph integrating:
     - Spatial proximity (from ST)  
     - Expression similarity (latent embeddings from **scANVI**)  
   - Applies a **graph attention network (GAT)** to propagate microenvironmental information.  
   - Learns pseudo microenvironment representations for scRNA-seq cells.

2. **Optimal Matching–based Cell Imputation**
   - Maps scRNA-seq cells to missing spatial locations using a **globally optimal matching strategy**.  
   - Preserves global tissue architecture and local microenvironment structures.

### Applications

The output of CellBack can be used to:  
- Recover missing cells in Slide-tags, SeekSpace, or similar platforms  
- Improve spatial clustering and substructure detection  
- Facilitate cell–cell communication analysis  
- Detect spatially variable genes  
- Integrate spatial RNA with single-cell multi-omics datasets to generate **spatial multi-omics profiles** at single-cell resolution

![CellBack overview](source/img/CellBack_framework.png)
