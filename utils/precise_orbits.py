"""
IGS precise orbits (SP3), as an independent check on the broadcast ephemeris.

The broadcast ephemeris is what a receiver actually has, so it is what the
navigation solution uses.  Precise orbits are what the truth turned out to be --
a few centimetres, computed after the fact from a global station network -- so the
difference between them is the part of the position error that arrived with the
signal rather than being made by the receiver.

`gnss_tools.orbits.sp3_utils` parses and interpolates SP3 correctly, and that half
is used here.  Its *download* half cannot work: `download_and_decompress_sp3_file`
calls `http_download` with no credentials at all, and CDDIS has required Earthdata
login for years.  So this module fetches the file itself -- through
`utils.cddis`, which handles the redirect that strips them -- and places it at
exactly the path `sp3_utils` looks in, so the upstream parser then finds it
already present and skips its own download.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gnss_tools.misc.data_utils import decompress, format_filepath

from . import cddis, environment_variables

# Two products, in preference order.  CODE's MGEX solution covers every
# constellation at 5-minute sampling; the IGS operational solution is
# GPS+GLONASS at 15 minutes and is a tenth the size.  Either is far better than
# broadcast for this purpose.
_PRODUCTS = (
    "gnss/products/{wwww}/COD0MGXFIN_{yyyy}{ddd}0000_01D_05M_ORB.SP3.gz",
    "gnss/products/{wwww}/IGS0OPSFIN_{yyyy}{ddd}0000_01D_15M_ORB.SP3.gz",
)

# Where `gnss_tools.orbits.sp3_utils` expects to find its cache.  Matching this
# exactly is the whole trick: put the file here and the upstream loader will not
# try to download it.
_SP3_CACHE_SUBDIR = "cddis"


def sp3_paths(day: datetime, resources_dir: str | Path, template: str) -> tuple[str, Path, Path]:
    """`(url, compressed_path, decompressed_path)` for one product on one day."""
    relative = format_filepath(template, day)
    # sp3_utils builds its own paths from a "gps/products/..." template rooted at
    # <resources>/cddis/, so mirror that layout whatever URL prefix is used.
    cache_relative = relative.replace("gnss/products", "gps/products", 1)
    compressed = Path(resources_dir) / _SP3_CACHE_SUBDIR / cache_relative
    return (
        f"{cddis.CDDIS_ARCHIVE_URL}/{relative}",
        compressed,
        compressed.with_suffix(""),
    )


def download_sp3(
    day: datetime,
    resources_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Fetch and decompress one day of precise orbits, returning the local path.

    Tries each product in turn.  A given day may be missing one of them -- CODE
    and IGS publish on different latencies, and older weeks use different naming
    -- so failing over is worth the few lines.
    """
    if resources_dir is None:
        resources_dir = environment_variables.get_resources_path()

    failures = []
    for template in _PRODUCTS:
        url, compressed, decompressed = sp3_paths(day, resources_dir, template)

        # A cache hit has to be a *usable* file, not merely a present one.  Two
        # kinds of residue get left behind by a failed Earthdata login, and both
        # otherwise fail forever with a gzip error that names no cause: an HTML
        # page written where the archive was expected, and the empty file that
        # decompressing it produces.  Treat either as a miss.
        if decompressed.exists() and decompressed.stat().st_size > 0 and not overwrite:
            return decompressed
        poisoned = compressed.exists() and compressed.read_bytes()[:2] != b"\x1f\x8b"
        if poisoned:
            decompressed.unlink(missing_ok=True)

        try:
            cddis.download(url, compressed, overwrite=overwrite or poisoned)
            decompress(str(compressed), str(decompressed))
            if decompressed.exists() and decompressed.stat().st_size > 0:
                return decompressed
            failures.append(f"{compressed.name}: decompressed to an empty file")
        except Exception as exc:
            failures.append(f"{compressed.name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"No precise orbit product available for {day.date()}. Tried:\n  "
        + "\n  ".join(failures)
    )


def load_sp3(day: datetime, resources_dir: str | Path | None = None):
    """
    `(epochs, records)` for one day, via `gnss_tools.orbits.parse_sp3`.

    The file is fetched here first so the upstream loader finds it cached.
    """
    from gnss_tools.orbits.parse_sp3 import parse_sp3_file

    path = download_sp3(day, resources_dir)
    return parse_sp3_file(str(path))
