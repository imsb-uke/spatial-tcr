"""CLI helpers for running project notebooks with papermill."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_NOTEBOOK_DIR = Path("notebooks/xenium-tcr-analysis")
DEFAULT_OUTPUT_DIR = Path(".papermill") / "xenium-tcr-analysis"
DEFAULT_KERNEL = "python3"
DEFAULT_SUBDIR_ORDER = ("vgene-analysis", "revision-extras")


def build_parser(
    *,
    description: str,
    default_notebook_dir: Path,
    default_output_dir: Path,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Run without asking for confirmation.",
    )
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=default_notebook_dir,
        help=(
            f"Directory containing notebooks to run (default: {default_notebook_dir})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=(
            "Directory where executed notebooks are written "
            f"(default: {default_output_dir})."
        ),
    )
    parser.add_argument(
        "--kernel",
        default=DEFAULT_KERNEL,
        help=f"Jupyter kernel name for papermill (default: {DEFAULT_KERNEL}).",
    )
    parser.add_argument(
        "--glob",
        default="**/*.ipynb",
        help="Notebook glob within --notebook-dir (default: **/*.ipynb).",
    )
    parser.add_argument(
        "--start-at",
        help="Start execution at the notebook with this filename or relative path.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="Start execution at the numbered notebook from the printed plan.",
    )
    return parser


def iter_notebooks(notebook_dir: Path, pattern: str) -> list[Path]:
    notebooks = [path for path in notebook_dir.glob(pattern) if path.is_file()]
    return sorted(notebooks, key=lambda path: notebook_sort_key(path, notebook_dir))


def run_notebook(
    notebook: Path, notebook_dir: Path, output_dir: Path, kernel: str
) -> None:
    output_path = build_output_path(notebook, notebook_dir, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "papermill",
        str(notebook),
        str(output_path),
        "-k",
        kernel,
    ]

    print(f"\n[{notebook.name}] Running", flush=True)
    subprocess.run(cmd, check=True)


def print_outline(notebooks: list[Path], notebook_dir: Path, output_dir: Path) -> None:
    print("Execution plan:", flush=True)
    for index, notebook in enumerate(notebooks, start=1):
        print(
            f"  {index:>2}. {format_notebook_path(notebook, notebook_dir)}", flush=True
        )

    print("", flush=True)
    print(f"Notebook directory: {notebook_dir}", flush=True)
    print(f"Output directory:   {output_dir}", flush=True)
    print(f"Notebook count:     {len(notebooks)}", flush=True)


def build_output_path(notebook: Path, notebook_dir: Path, output_dir: Path) -> Path:
    try:
        relative_path = notebook.relative_to(notebook_dir)
    except ValueError:
        relative_path = Path(notebook.name)

    return output_dir / relative_path


def format_notebook_path(notebook: Path, notebook_dir: Path) -> Path:
    try:
        return notebook.relative_to(notebook_dir)
    except ValueError:
        return notebook


def notebook_sort_key(
    notebook: Path, notebook_dir: Path
) -> tuple[int, str, tuple[str, ...]]:
    relative_path = format_notebook_path(notebook, notebook_dir)
    parts = relative_path.parts
    if len(parts) == 1:
        return (0, "", parts)

    top_level_dir = parts[0]
    try:
        bucket = DEFAULT_SUBDIR_ORDER.index(top_level_dir) + 1
    except ValueError:
        bucket = len(DEFAULT_SUBDIR_ORDER) + 1

    return (bucket, top_level_dir, parts[1:])


def find_start_index(
    notebooks: list[Path],
    notebook_dir: Path,
    start_at: str | None,
    start_index: int | None,
) -> int:
    if start_at is not None and start_index is not None:
        raise ValueError("Use only one of --start-at or --start-index.")

    if start_at is not None:
        for index, notebook in enumerate(notebooks):
            relative_path = str(format_notebook_path(notebook, notebook_dir))
            if notebook.name == start_at or relative_path == start_at:
                return index
        raise ValueError(f"Notebook not found for --start-at: {start_at}")

    if start_index is not None:
        if 1 <= start_index <= len(notebooks):
            return start_index - 1
        raise ValueError(
            f"--start-index must be between 1 and {len(notebooks)}; got {start_index}."
        )

    return prompt_start_index(notebooks)


def prompt_start_index(notebooks: list[Path]) -> int:
    response = input(f"Start from notebook [1-{len(notebooks)}, default 1]: ").strip()
    if not response:
        return 0

    try:
        index = int(response)
    except ValueError as error:
        raise ValueError("Start index must be a number.") from error

    if 1 <= index <= len(notebooks):
        return index - 1

    raise ValueError(f"Start index must be between 1 and {len(notebooks)}.")


def confirm_run() -> bool:
    response = input("Proceed with notebook execution? [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def main(
    *,
    description: str = "Run Xenium TCR analysis notebooks sequentially with papermill.",
    default_notebook_dir: Path = DEFAULT_NOTEBOOK_DIR,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    argv: list[str] | None = None,
) -> int:
    args = build_parser(
        description=description,
        default_notebook_dir=default_notebook_dir,
        default_output_dir=default_output_dir,
    ).parse_args(argv)
    notebook_dir = args.notebook_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not notebook_dir.is_dir():
        print(f"Notebook directory does not exist: {notebook_dir}", file=sys.stderr)
        return 1

    notebooks = iter_notebooks(notebook_dir, args.glob)
    if not notebooks:
        print(
            f"No notebooks matched '{args.glob}' in {notebook_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print_outline(notebooks, notebook_dir, output_dir)
    print(f"Kernel:             {args.kernel}", flush=True)

    try:
        start_index = find_start_index(
            notebooks,
            notebook_dir,
            args.start_at,
            args.start_index,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    notebooks_to_run = notebooks[start_index:]

    if not args.yes and not confirm_run():
        print("Aborted.", flush=True)
        return 1

    for notebook in notebooks_to_run:
        run_notebook(notebook, notebook_dir, output_dir, args.kernel)

    print("\nAll notebooks completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
