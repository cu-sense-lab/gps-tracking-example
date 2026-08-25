"""
Downloading from CDDIS, through NASA's Earthdata login.

One function, and it exists because the obvious way does not work.

CDDIS redirects to `urs.earthdata.nasa.gov` to authenticate, and `requests`
deliberately strips credentials when a redirect crosses hosts -- a security
feature, not a bug.  So a one-shot `requests.get(url, auth=...)` follows the
redirect, arrives unauthenticated, and returns **200 OK with a login page**.  The
failure then surfaces much later as `BadGzipFile: Not a gzipped file (b'<!')`,
which points nowhere near the cause.

Both of `gnss_tools`' download paths do the one-shot version --
`misc.data_utils.http_download` passes auth once and lets it be stripped, and
`orbits.sp3_utils.download_and_decompress_sp3_file` calls it with no auth at all.
Neither can fetch from CDDIS today.

The documented pattern is two hops on one session: let the first request end up at
the URS authorize URL, then request *that* URL with the credentials attached.
"""

from __future__ import annotations

from pathlib import Path

import requests

from . import environment_variables

CDDIS_ARCHIVE_URL = "https://cddis.nasa.gov/archive"


def credentials() -> tuple[str, str]:
    """Earthdata username and password, or a message saying how to get them."""
    username, password = environment_variables.get_earthdata_credentials()
    if not username or not password:
        raise RuntimeError(
            "Downloading from CDDIS needs Earthdata credentials. Set "
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD in .env -- registration is "
            "free at https://urs.earthdata.nasa.gov/, and the account must also "
            "accept the CDDIS EULA before downloads succeed."
        )
    return username, password


def download(url: str, destination: str | Path, *, overwrite: bool = False) -> Path:
    """
    Fetch one file from CDDIS to `destination`, creating parent directories.

    Cached: an existing file is left alone unless `overwrite`, because a notebook
    re-run should not re-download a hundred megabytes of orbit products.
    """
    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination

    username, password = credentials()
    session = requests.Session()
    session.auth = (username, password)

    # Two hops.  See the module docstring for why one is not enough.
    redirected = session.request("get", url)
    response = session.get(redirected.url, auth=(username, password))

    if response.status_code != 200:
        raise RuntimeError(
            f"CDDIS returned {response.status_code} for {url}. A 401 means the "
            "Earthdata credentials in .env are wrong; a 404 means the product is "
            "not published under that name for this date."
        )
    if response.content[:1] == b"<":
        raise RuntimeError(
            f"CDDIS returned an HTML page rather than a file for {url}. That is "
            "an Earthdata login that did not complete -- check the credentials in "
            ".env, and that the account has accepted the CDDIS EULA."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination
