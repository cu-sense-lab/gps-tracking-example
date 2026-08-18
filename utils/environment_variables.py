"""
Runtime configuration loaded from a `.env` file at the repository root (see the
template in README.md). Importing this module loads `.env` into the process
environment, so `gnss_tools`'s own `os.environ.get(...)` reads (e.g.
`cddis_download_utils.py`'s Earthdata credentials) pick the values up too --
notebooks and scripts don't need to call `load_dotenv()` themselves.

Each environment variable has a corresponding `get_*()` accessor below,
documented with what the variable controls. Directory-valued variables are
resolved to a `Path` and checked for existence at call time; a missing
directory emits a `warnings.warn` rather than raising, since the caller is
often about to create it (e.g. a fresh `local-data/`). A variable that has no
usable default and is simply unset does still raise -- there's no path to
warn about in that case.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

dotenv.load_dotenv(REPO_ROOT / ".env")

_PROCESSING_PATH: Optional[str] = os.environ.get("PROCESSING_PATH") or None
_DATA_DIR: Optional[str] = os.environ.get("DATA_DIR") or None
_COLLECTS_DIR: Optional[str] = os.environ.get("COLLECTS_DIR") or None
_EARTHDATA_USERNAME: Optional[str] = os.environ.get("EARTHDATA_USERNAME") or None
_EARTHDATA_PASSWORD: Optional[str] = os.environ.get("EARTHDATA_PASSWORD") or None


def _resolve_dir(env_var: str, value: Optional[str], *, default: Optional[Path] = None) -> Path:
    """Resolve `env_var`'s value to a `Path`, warning (not raising) if it's missing on disk."""
    if value:
        path = Path(value)
    elif default is not None:
        path = default
    else:
        raise ValueError(f"{env_var} is not set (see the `.env` template in README.md).")
    if not path.exists():
        warnings.warn(f"{env_var} points to '{path}', but that directory does not exist.")
    return path


def get_data_dir() -> Path:
    """
    `DATA_DIR`: base data directory used by `gnss_tools` for downloaded
    RINEX/orbit (SP3) data (`gnss_tools.rinex_io.cddis_download_utils`,
    `gnss_tools.misc.rinex_obs_utils`). No local default -- those helpers
    fall back to the current working directory if it's unset, which is
    rarely what you want, so this raises instead of guessing a path.
    """
    return _resolve_dir("DATA_DIR", _DATA_DIR)


def get_collects_dir() -> Path:
    """
    `COLLECTS_DIR`: directory containing GNSS raw data collects, one
    subdirectory per experiment (`<experiment_name>/metadata.yml` plus the
    raw IQ files it references). No local default; raises if unset.
    """
    return _resolve_dir("COLLECTS_DIR", _COLLECTS_DIR)


def get_earthdata_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`: credentials for downloading
    RINEX/orbit (SP3) data from CDDIS/Earthdata
    (`gnss_tools.rinex_io.cddis_download_utils` / `earthdata_utils`). Returns
    `(username, password)`, either of which may be `None` if unset -- callers
    that need both for HTTP basic auth should check before using them.
    """
    if _EARTHDATA_USERNAME is None or _EARTHDATA_PASSWORD is None:
        warnings.warn("EARTHDATA_USERNAME and/or EARTHDATA_PASSWORD are not set.")
    return _EARTHDATA_USERNAME, _EARTHDATA_PASSWORD
