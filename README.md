# Spatial Analysis of T Cell Clonality in Autoimmune Kidney Disease Using TRV Probes

This repository contains code and analysis notebooks for the manuscript **Spatial Analysis of T Cell Clonality in Autoimmune Kidney Disease Using TRV Probes**.

## Installation

Prerequisites: Git installed.

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Download the code and install dependencies

```bash
git clone https://github.com/imsb-uke/spatial-tcr.git
cd spatial-tcr
uv sync
```

## Usage

The `notebooks` folder contains separate analysis folders:

- The `XeniumTCR-analysis` folder contains the notebooks for the spatial analysis of the Xenium TCR data.
- The `TCRseq-analysis` folder contains code to analyze the single-cell TCRseq data.
