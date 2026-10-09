"""Export a Jupyter notebook to a clean, VS Code-friendly Python file."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from nbconvert import PythonExporter

GENERATED_PREAMBLE = "# This file was generated from a Jupyter notebook.\n"
RUFF_NOTEBOOK_EXCEPTION = "# ruff: noqa: B018\n"


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
        raise SystemExit(f"Output already exists: {output}\nUse --force to overwrite it.")

    exporter = PythonExporter()
    source, _ = exporter.from_filename(str(notebook))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(clean_export(source), encoding="utf-8")
    run_ruff(output)

    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
