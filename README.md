# Spatial Analysis of T Cell Clonality in Autoimmune Kidney Disease Using TRV Probes

This repository contains code and analysis notebooks for the manuscript **Spatial Analysis of T Cell Clonality in Autoimmune Kidney Disease Using TRV Probes**. The project combines spatial transcriptomics (Xenium) with TCR sequencing to understand the distribution and clonal relationships of T cells within kidney tissue.

## Installation (Conda + Poetry)
Prerequisites: Conda (Mambaforge/Miniconda/Anaconda) and Git installed.

1) Create and activate a Python 3.10 environment

```bash
conda create -n spatialtcr python=3.10 -y
conda activate spatialtcr
```

2) Install Poetry inside the active Conda environment (if not present)

```bash
conda install -c conda-forge poetry -y
# Make Poetry use the currently active Conda Python (recommended)
poetry config virtualenvs.prefer-active-python true
```

3) Download code and install dependencies

```bash
git clone https://github.com/imsb-uke/spatial-tcr.git
cd spatial-tcr
poetry install
```

## Usage

The `notebooks` folder contains separate analysis folders:
- The `XeniumTCR-analysis` folder contains the notebooks for the spatial analysis of the Xenium TCR data.
- The `TCRseq-analysis` folder contains code to analyze the single-cell TCRseq data.