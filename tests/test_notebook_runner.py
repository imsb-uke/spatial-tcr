from pathlib import Path

from spatial_tcr.notebook_runner import (
    build_output_path,
    build_parser,
    find_start_index,
    iter_notebooks,
)
from spatial_tcr.spatialtcr_runner import build_parser as build_spatialtcr_parser
from spatial_tcr.tcr_seq_notebook_runner import (
    DEFAULT_NOTEBOOK_DIR as TCR_SEQ_NOTEBOOK_DIR,
)
from spatial_tcr.tcr_seq_notebook_runner import DEFAULT_OUTPUT_DIR as TCR_SEQ_OUTPUT_DIR


def test_iter_notebooks_orders_main_then_vgene_then_revision(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()

    for relative_path in (
        "03-main.ipynb",
        "01-main.ipynb",
        "vgene-analysis/02-vgene.ipynb",
        "vgene-analysis/01-vgene.ipynb",
        "revision-extras/02-revision.ipynb",
        "revision-extras/01-revision.ipynb",
    ):
        path = notebook_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    notebooks = iter_notebooks(notebook_dir, "**/*.ipynb")

    assert [path.relative_to(notebook_dir) for path in notebooks] == [
        Path("01-main.ipynb"),
        Path("03-main.ipynb"),
        Path("vgene-analysis/01-vgene.ipynb"),
        Path("vgene-analysis/02-vgene.ipynb"),
        Path("revision-extras/01-revision.ipynb"),
        Path("revision-extras/02-revision.ipynb"),
    ]


def test_build_output_path_preserves_subdirectories(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks"
    output_dir = tmp_path / ".papermill"
    notebook = notebook_dir / "revision-extras" / "01-extra.ipynb"

    expected = output_dir / "revision-extras" / "01-extra.ipynb"

    assert build_output_path(notebook, notebook_dir, output_dir) == expected


def test_find_start_index_matches_relative_path_for_nested_notebook(
    tmp_path: Path,
) -> None:
    notebook_dir = tmp_path / "notebooks"
    notebooks = [
        notebook_dir / "01-main.ipynb",
        notebook_dir / "vgene-analysis" / "01-vgene.ipynb",
        notebook_dir / "revision-extras" / "01-extra.ipynb",
    ]

    index = find_start_index(
        notebooks,
        notebook_dir,
        "vgene-analysis/01-vgene.ipynb",
        None,
    )

    assert index == 1


def test_build_parser_allows_tcr_seq_defaults() -> None:
    parser = build_parser(
        description="Run TCR-seq analysis notebooks sequentially with papermill.",
        default_notebook_dir=TCR_SEQ_NOTEBOOK_DIR,
        default_output_dir=TCR_SEQ_OUTPUT_DIR,
    )

    args = parser.parse_args([])

    assert args.notebook_dir == TCR_SEQ_NOTEBOOK_DIR
    assert args.output_dir == TCR_SEQ_OUTPUT_DIR


def test_build_spatialtcr_parser_defaults() -> None:
    args = build_spatialtcr_parser().parse_args([])

    assert args.kernel == "python3"
    assert args.yes is False
