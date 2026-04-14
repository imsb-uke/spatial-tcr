"""Entry point for running all Spatial TCR notebook pipelines."""

from __future__ import annotations

import argparse

from spatial_tcr.notebook_runner import main as run_xenium_notebooks
from spatial_tcr.tcr_seq_notebook_runner import main as run_tcr_seq_notebooks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Xenium and TCR-seq notebook pipelines sequentially."
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name for papermill in both pipelines (default: python3).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    xenium_argv: list[str] = ["--kernel", args.kernel]

    xenium_status = run_xenium_notebooks(argv=xenium_argv)
    if xenium_status != 0:
        return xenium_status

    tcr_seq_argv: list[str] = ["--kernel", args.kernel]

    return run_tcr_seq_notebooks(argv=tcr_seq_argv)


if __name__ == "__main__":
    raise SystemExit(main())
