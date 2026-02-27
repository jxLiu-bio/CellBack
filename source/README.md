Spatial transcriptomics (ST) enables gene expression profiling within intact tissues while preserving spatial context. However, recently developed **single-cell whole-transcriptome spatial transcriptomics (scWT-ST) platforms** (e.g. *Slide-tags*, *SeekSpace*) often suffer from **substantial cell loss**, resulting in incomplete tissue landscapes and compromised downstream analyses.
**CellBack** is a **microenvironment-aware cell imputation framework** that reconstructs complete spatial tissue maps at **single-cell resolution** by optimally mapping cells from matched scRNA-seq data into spatial contexts from a global perspective.  
By integrating **learned microenvironment representations** with a **globally optimal matching strategy**, CellBack faithfully recovers missing cells while preserving spatial tissue architectures.

In addition to cell imputation, CellBack:
- predicts transcript distributions for imaging-based ST platforms  
- denoises gene expression for sequencing-based ST platforms  
- enables spatial multi-omics integration at single-cell resolution  
