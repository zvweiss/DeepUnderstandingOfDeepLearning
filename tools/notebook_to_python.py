"""Export a Jupyter notebook to a clean, VS Code-friendly Python file."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from nbconvert import PythonExporter

GENERATED_PREAMBLE = "# This file was generated from a Jupyter notebook.\n"
RUFF_NOTEBOOK_EXCEPTION = "# ruff: noqa: B018, I001\n"
NOTEBOOK_DISPLAY_SETUP = re.compile(
    r"^\s*(?:%matplotlib\s+inline|get_ipython\(\)\.run_line_magic\("
    r"[\"']matplotlib[\"'],\s*[\"']inline[\"']\)|"
    r"import\s+matplotlib_inline\.backend_inline|"
    r"matplotlib_inline\.backend_inline\.set_matplotlib_formats\([^)]*\))\s*$"
)
NOTEBOOK_ONLY_CODE = (
    (re.compile(r"\bgoogle\.colab\b"), "Google Colab import"),
    (re.compile(r"\bfiles\.upload\s*\("), "Google Colab file upload"),
    (re.compile(r"\bdrive\.mount\s*\("), "Google Drive mount"),
    (re.compile(r"\bget_ipython\s*\("), "IPython command"),
)


class NotebookOnlyCodeError(ValueError):
    """Raised when an export needs a user-selected local replacement."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a notebook to formatted, debuggable Python code."
    )
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: notebook path with a .py suffix)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file",
    )
    return parser.parse_args()


def clean_export(source: str) -> str:
    """Remove nbconvert boilerplate and retain notebook cells as VS Code cells."""
    lines = source.splitlines()

    while lines and (
        lines[0].startswith("#!")
        or re.fullmatch(r"#.*coding[:=]\s*[-\w.]+", lines[0]) is not None
        or not lines[0].strip()
    ):
        lines.pop(0)

    # Matplotlib displays figures without this Jupyter-only directive in a normal
    # Python process (with plt.show()) or in VS Code's interactive window.
    lines = [line for line in lines if NOTEBOOK_DISPLAY_SETUP.fullmatch(line) is None]

    unsupported: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        for pattern, description in NOTEBOOK_ONLY_CODE:
            if pattern.search(line):
                unsupported.append(f"line {line_number}: {description}: {line.strip()}")
                break

    if unsupported:
        details = "\n".join(f"  - {item}" for item in unsupported)
        raise NotebookOnlyCodeError(
            "Cannot create a standalone .py file until this notebook-only code "
            f"is replaced:\n{details}\n"
            "Use a local file path (pathlib.Path or a file-picker library) in place "
            "of Colab upload/mount helpers."
        )

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"^# In\[[^]]*\]:\s*$", "# %%", cleaned, flags=re.MULTILINE)
    return f"{GENERATED_PREAMBLE}{RUFF_NOTEBOOK_EXCEPTION}\n{cleaned.rstrip()}\n"


def run_ruff(output: Path) -> None:
    """Apply lint modernization and formatting with the environment's Ruff."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--fix",
            "--unsafe-fixes",
            str(output),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(output)],
        check=True,
    )


def main() -> int:
    args = parse_args()
    notebook = args.notebook.resolve()
    output = (args.output or notebook.with_suffix(".py")).resolve()

    if notebook.suffix != ".ipynb":
        raise SystemExit(f"Expected a .ipynb file: {notebook}")
    if not notebook.is_file():
        raise SystemExit(f"Notebook not found: {notebook}")
    if output.exists() and not args.force:
        raise SystemExit(
            f"Output already exists: {output}\nUse --force to overwrite it."
        )

    exporter = PythonExporter()
    source, _ = exporter.from_filename(str(notebook))
    try:
        source = clean_export(source)
    except NotebookOnlyCodeError as error:
        raise SystemExit(f"Could not export {notebook}:\n{error}") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    run_ruff(output)

    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
