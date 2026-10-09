# Deep Understanding Of Deep Learning
Python code accompanying the course "A deep understanding of deep learning (with Python intro)"

Master deep learning in PyTorch using an experimental scientific approach, with lots of examples and practice problems.


See https://www.udemy.com/course/deeplearning_x/?couponCode=202502 for more details, preview videos, and to enroll in the full course.

## Local Python setup

This checkout uses a repository-local virtual environment. From the repository
root, activate it with:

```bash
source .venv/bin/activate
```

VS Code is configured to use `.venv/bin/python` for Python files and notebooks.
If VS Code prompts for a notebook kernel, choose the interpreter from this
repository's `.venv` directory.

To run a Python file normally:

```bash
python path/to/file.py
```

To debug an extracted Python file, open it in VS Code, add breakpoints, select
`Python: Current File` in the Run and Debug view, and press F5. The debugger
runs from the repository root so paths such as `data/...` resolve consistently.

## Exporting notebooks to Python

The project uses [Poe the Poet](https://poethepoet.natn.io/) for short,
documented commands. After activating `.venv`, export any notebook from the
repository root with:

```bash
poe export path/to/notebook.ipynb
```

For example:

```bash
poe export math/DUDL_math_argmin.ipynb
```

The generated `.py` file is placed beside its notebook. In this example, Poe
creates `math/DUDL_math_argmin.py` and leaves the original notebook unchanged.

The converter removes the generated shebang and redundant UTF-8 declaration,
changes notebook cell comments to VS Code `# %%` cell markers, modernizes code
where Ruff can do so automatically, and formats the result.

To protect edits made after exporting, the command will not overwrite an
existing `.py` file. To deliberately regenerate it from the notebook, use:

```bash
poe export math/DUDL_math_argmin.ipynb --force
```

Run `poe` to list the available project tasks or `poe --help export` to see the
export command's arguments.

VS Code also supports Python cell markers. Add `# %%` before a section of a
`.py` file to run that section interactively while keeping the file directly
executable and debuggable.

Notebook-only commands such as `%matplotlib inline` and Google Colab helpers
must be removed or replaced when notebook cells are moved into `.py` files.

To rebuild the environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
