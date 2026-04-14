"""Entry point for running TCR-seq analysis notebooks."""

from pathlib import Path

from spatial_tcr.notebook_runner import main as run_notebooks

DEFAULT_NOTEBOOK_DIR = Path("notebooks/tcr-seq-analysis")
DEFAULT_OUTPUT_DIR = Path(".papermill") / "tcr-seq-analysis"


def main(argv: list[str] | None = None) -> int:
    return run_notebooks(
        description="Run TCR-seq analysis notebooks sequentially with papermill.",
        default_notebook_dir=DEFAULT_NOTEBOOK_DIR,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
