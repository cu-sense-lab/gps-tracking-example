
## Installation

If you know what you're doing and want to use your own virtual environment manager (`uv`, `venv`, etc.), go for it!

Personally, I use conda, and the following instructions assume it is installed on your system.
If you don't have conda, you may download miniforge, located here: https://github.com/conda-forge/miniforge

It also assumes you have the `poetry` package manager installed.  (I do this because I find its dependency resolution to be better..)
See install here:  https://python-poetry.org/docs/

To set up the environment, you can run:

```bash
conda env create -f environment.yml
```

To activate the environment, run:

```bash
conda activate gnss_tracking_example
```

To install the package in editable mode, run:

```bash
git submodule update --init --recursive
poetry install
```

Note: there is a submodule `gnss-tools` in this repository that contains some utility functions.  It is another github repository, located here:
https://github.com/cu-sense-lab/gnss-tools

## Usage

You can add your own utilities/functions to the `utils/` folder as needed.

Please email me if you have any problems, and I will do my best to help!


## Updating Environment

*Aside*: I had forgotten that `numba` (a package for JIT compiling Python code) is not
compatible with Python 3.14 yet, so I had to downgrade back to 3.13.  If you already made
a Python 3.14 environment, you can remake it with:

    conda env remove -n gnss_lectures
    conda env create -f environment.yml
    conda activate gnss_lectures
    poetry install