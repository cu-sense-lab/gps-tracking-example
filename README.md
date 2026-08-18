
# GNSS Processing Package

The current (as of 2026) SeNSe Lab GNSS processing package is a collection of Python scripts and functions that can be used to process GNSS data.  It is designed to be modular, so that you can use only the parts you need for your own research or learning. 

## Getting Started

1. Clone the repository and initialize submodules:

```bash
git clone https://github.com/cu-sense-lab/gps-tracking-example
cd gps-tracking-example
git submodule update --init --recursive
```

2. Conda+Poetry Environment Setup

If you know what you're doing and want to use your own virtual environment manager (`uv`, `venv`, etc.), go for it!

Personally, I use conda, and the following instructions assume it is installed on your system.
If you don't have conda, you may download miniforge, located here: https://github.com/conda-forge/miniforge

It also assumes you have the `poetry` package manager installed.  (I do this because I find its dependency resolution to be better..)
See install here:  https://python-poetry.org/docs/

To set up the environment, you can run:

```bash
conda env create -f environment.yml --prefix ./.conda_env
```

To activate the environment, run (from the root of this repository):

```bash
conda activate ./.conda_env
```

To install the packages, run:

```bash
poetry install
```

3. Environment Variable Setup

Runtime configurations (data locations and credentials) are read from a `.env` file in the root of this repository.
Copy the template below and fill in real values for your machine. Each variable has a
corresponding `get_*()` accessor in `utils/environment_variables.py` (e.g. `get_outputs_path()`,
`get_resources_path()`, `get_collects_path()`, `get_earthdata_credentials()`) — use those instead of
reading the environment directly.

```bash
# Copy the following to `.env` and fill in real values for your machine.
# `.env` is gitignored — never commit real paths/credentials.

# Your working directory for processing outputs
# (acquisition/tracking results, logs, etc.). See
# utils/environment_variables.py. Defaults to `<repo_root>/local-data` if unset.
OUTPUTS_PATH=

# Base data directory used by gnss-tools for downloaded RINEX/orbit (SP3) data.
RESOURCES_PATH=

# Path to directory containing GNSS raw data collects:
# <experiment_name>/{collect_metadata.yml, <collect_id>.<ext>}
COLLECTS_PATH=

# Earthdata credentials for downloading RINEX/orbit (SP3) data from CDDIS/Earthdata.
# helpers (gnss_tools.rinex_io.cddis_download_utils / earthdata_utils).
EARTHDATA_USERNAME=
EARTHDATA_PASSWORD=
```


## Notes

- There is a submodule `gnss-tools` in this repository that contains some utility functions.  It is another github repository, located here:
https://github.com/cu-sense-lab/gnss-tools

- You can add your own utilities/functions to the `utils/` folder as needed.

- Please email me if you have any problems, and I will do my best to help!

*Aside*: I had forgotten that `numba` (a package for JIT compiling Python code) is not
currently compatible with Python 3.14 yet, so I had to downgrade back to 3.13.  If you already made
a Python 3.14 environment, you can remake it with:

    conda env remove -n gnss_lectures
    conda env create -f environment.yml
    conda activate gnss_lectures
    poetry install


