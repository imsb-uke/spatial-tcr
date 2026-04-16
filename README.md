# Spatial Analysis of T Cell Clonality in Autoimmune Kidney Disease Using TRV Probes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19607985.svg)](https://doi.org/10.5281/zenodo.19607985)

This repository contains code and analysis notebooks for the manuscript **Spatial Analysis of T Cell Clonality in Autoimmune Kidney Disease Using TRV Probes**. The preprint can be found on [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.29.673064v1).

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

- The `xenium-tcr-analysis` folder contains the notebooks for the spatial analysis of the Xenium TCR data.
- The `tcr-seq-analysis` folder contains code to analyze the single-cell TCRseq data.

After `uv sync`, the package installs CLI commands that execute notebooks with [papermill](https://github.com/nteract/papermill) (executed copies go under `.papermill/` by default):

| Command | Purpose |
| --- | --- |
| `uv run xenium-notebooks` | Run the Xenium pipeline (`notebooks/xenium-tcr-analysis/`, output `.papermill/xenium-tcr-analysis/`). |
| `uv run tcr-seq-notebooks` | Run the TCR-seq pipeline (`notebooks/tcr-seq-analysis/`, output `.papermill/tcr-seq-analysis/`). |
| `uv run spatialtcr` | Run the Xenium pipeline, then the TCR-seq pipeline (shared `--kernel`). |

Use `--help` on each command for options (e.g. `--notebook-dir`, `--output-dir`, `--start-at` or `--start-index` (mutually exclusive), `--glob` on the first two).

Adapt **raw or external input** paths where notebooks hardcode them (leave `data/**/processed` paths as in the repo unless you intentionally relocate those objects):

- **TCR-seq:** `notebooks/tcr-seq-analysis/01-prepare-tcr-seq-data.ipynb` — set `ref_dir` to the directory that contains the reference metadata CSVs and the `TCRseq/...` inputs (outputs are written under `data/tcr-seq/processed/`).
- **Xenium (raw run / morphology):** `notebooks/xenium-tcr-analysis/00-check-transcript-assign.ipynb` and `01-prep_data.ipynb` — set `base_dir` / `data_dir` to your Xenium output folder (the parent of `output-*` region directories). The same `base_dir` pattern appears in notebooks that reload morphology for figures: `10.1-plot-nhood-zoom-in-v2.ipynb`, `10.3-plot-glom-area.ipynb`, `10.4-plot-dgT-abT-example.ipynb`, `13.1-gdT-spatial-plot.ipynb`, and `notebooks/xenium-tcr-analysis/revision-extras/14-plot-more-glom-examples.ipynb`.
- **Reference atlas (classification):** `notebooks/xenium-tcr-analysis/03.1-classify-celltypes.ipynb` — the path to the Lake kidney reference `cellxgene_obj.h5ad` must point to your copy of that file.

Downstream notebooks that only read `data/xenium/processed/*.h5ad` or `data/tcr-seq/processed/*` do not need those raw paths changed.

## Citation

If you use this repository or the associated manuscript, please cite:

```bibtex
@article{ly2025spatial,
  title={Spatial Analysis of T Cell Clonality in Autoimmune Kidney Disease Using TRV Probes},
  author={Ly, Cedric and Schaub, Darius P and Khatri, Robin and Sultana, Zeba and Boxnick, Annika and Song, Zheng and Huber, Tobias and Wiech, Thorsten and Tolosa, Eva and Panzer, Ulf and others},
  journal={bioRxiv},
  pages={2025--08},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
