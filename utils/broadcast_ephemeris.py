"""
Fetching and choosing broadcast ephemerides.

This exists because the obvious upstream route does not work.
`gnss_tools.misc.rinex_gps_ephemeris` is the module that would do this, and it
cannot even be imported: it does `from ..rinex.rinex_nav import ...` against a
package that is called `rinex_io`.  Its `compute_gps_satellite_pvt_from_ephemeris`
is separately broken, calling both of its own helpers with the wrong signatures.
Both are bugs in a sibling repository, not something to work around in place, so
this module goes to the pieces underneath them instead: `utils.cddis` for the
download (`gnss_tools`' own downloader cannot authenticate -- see there) and
`gnss_tools.rinex_io.rinex_nav.parse_RINEX_LNAV_data` for the parse, which is
fine even though the file-level wrapper around it is not.

What is here is only the two things a navigation solution needs: get the daily
`brdc` file for a date, and pick the right record out of it for a given satellite
and time.  Choosing the record is not a formality -- a daily file holds a couple of
dozen ephemerides per satellite and they are not interchangeable.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from gnss_tools.misc.data_utils import decompress, format_filepath
from gnss_tools.rinex_io.rinex_nav import RINEX_LNAVEphemeris, parse_RINEX_LNAV_data

from . import cddis, environment_variables
from .nav.ephemeris import LnavEphemeris, from_rinex

# CDDIS serves the daily broadcast file over HTTPS behind Earthdata login.  The
# compression changed from .Z to .gz at the end of 2020.
_BRDC_DIRECTORY = "/gnss/data/daily/{yyyy}/{ddd}/{yy}n/"
_BRDC_FILENAME_GZ = "brdc{ddd}0.{yy}n.gz"
_BRDC_FILENAME_Z = "brdc{ddd}0.{yy}n.Z"
_COMPRESSION_CHANGE = datetime(2020, 12, 1)

# Beyond this far from its reference time an ephemeris is being extrapolated well
# outside its fit interval.  The nominal LNAV fit is four hours centred on toe.
MAX_AGE_S = 7200.0


def brdc_url_and_path(day: datetime, resources_dir: str | Path) -> tuple[str, Path, Path]:
    """`(url, compressed_path, decompressed_path)` for one day's broadcast file."""
    filename = _BRDC_FILENAME_Z if day < _COMPRESSION_CHANGE else _BRDC_FILENAME_GZ
    relative = format_filepath(_BRDC_DIRECTORY + filename, day)
    url = cddis.CDDIS_ARCHIVE_URL + relative
    compressed = Path(resources_dir) / relative.lstrip("/")
    return url, compressed, compressed.with_suffix("")


def download_brdc(
    day: datetime, resources_dir: str | Path | None = None, *, overwrite: bool = False
) -> Path:
    """
    Fetch and decompress one day's `brdc` file, returning the local path.

    Cached: an already-decompressed file is returned untouched, which matters
    because a notebook re-run should not re-download.
    """
    if resources_dir is None:
        resources_dir = environment_variables.get_resources_path()
    url, compressed, decompressed = brdc_url_and_path(day, resources_dir)

    if decompressed.exists() and not overwrite:
        return decompressed

    # An HTML page cached where the archive belongs is the residue of a failed
    # Earthdata login; it would otherwise fail forever with a gzip error naming no
    # cause.  See `utils.cddis` for why that happens at all.
    poisoned = compressed.exists() and compressed.read_bytes()[:2] != b"\x1f\x8b"
    cddis.download(url, compressed, overwrite=overwrite or poisoned)
    decompress(str(compressed), str(decompressed))
    return decompressed


def _split_header(lines: list[str]) -> tuple[list[str], list[str]]:
    """`(header_lines, data_lines)`, split at END OF HEADER."""
    for i, line in enumerate(lines):
        if "END OF HEADER" in line:
            return lines[: i + 1], lines[i + 1 :]
    return [], lines


def _parse_fortran_float(text: str) -> float:
    """RINEX writes exponents with `D`, which Python does not accept."""
    return float(text.strip().replace("D", "E").replace("d", "e"))


def parse_iono_header(lines: list[str]) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """
    The Klobuchar coefficients from a RINEX 2 navigation header, if present.

    A useful fallback: the same eight numbers arrive in CNAV message type 30, but a
    short tracking run may not have caught one, and an L5 fix without an
    ionospheric correction is several metres out -- mostly in height.

    Returns None when the header carries no ION ALPHA/ION BETA lines, which is
    normal for a RINEX 3 file (they are labelled differently there).
    """
    alpha = beta = None
    for line in lines:
        label = line[60:].strip()
        if label == "ION ALPHA":
            alpha = tuple(_parse_fortran_float(line[2 + 12 * k : 2 + 12 * (k + 1)]) for k in range(4))
        elif label == "ION BETA":
            beta = tuple(_parse_fortran_float(line[2 + 12 * k : 2 + 12 * (k + 1)]) for k in range(4))
    if alpha is None or beta is None:
        return None
    return alpha, beta


def load_brdc(
    day: datetime, resources_dir: str | Path | None = None
) -> dict[int, list[RINEX_LNAVEphemeris]]:
    """
    Every ephemeris record in one day's file, keyed by PRN.

    Calls `parse_RINEX_LNAV_data` on the body rather than
    `parse_RINEX_LNAV_file` on the whole thing.  The file-level function hands its
    header lines to `rinex2.parse_header`, which expects a file object and calls
    `.readline()` on the list -- so it raises `AttributeError` on every file.  The
    body parser underneath it is fine, and the header is parsed here instead for
    the one thing in it that matters (see `parse_iono_header`).
    """
    with open(download_brdc(day, resources_dir), "r") as f:
        lines = f.readlines()
    _, data_lines = _split_header(lines)
    return parse_RINEX_LNAV_data(data_lines)


def load_brdc_iono(
    day: datetime, resources_dir: str | Path | None = None
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """The broadcast Klobuchar coefficients for one day, or None."""
    with open(download_brdc(day, resources_dir), "r") as f:
        lines = f.readlines()
    header, _ = _split_header(lines)
    return parse_iono_header(header)


def load_brdc_span(
    start: datetime, end: datetime, resources_dir: str | Path | None = None
) -> dict[int, list[RINEX_LNAVEphemeris]]:
    """
    Records covering a time span, merging whole days.

    The day either side is included because a collect near midnight needs
    ephemerides whose fit interval straddles the boundary, and because the record
    valid at 00:05 was usually uploaded the previous day.
    """
    records: dict[int, list[RINEX_LNAVEphemeris]] = {}
    day = start.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while day <= last:
        try:
            for prn, day_records in load_brdc(day, resources_dir).items():
                records.setdefault(prn, []).extend(day_records)
        except Exception as exc:  # a missing neighbouring day is not fatal
            if start.date() <= day.date() <= end.date():
                raise
            print(f"  (skipping {day.date()}: {type(exc).__name__}: {exc})")
        day += timedelta(days=1)
    return records


def select_ephemeris(
    records: list[RINEX_LNAVEphemeris],
    tow_s: float,
    week: int,
    *,
    max_age_s: float = MAX_AGE_S,
    require_healthy: bool = True,
) -> RINEX_LNAVEphemeris | None:
    """
    The record to use for one satellite at one time, or None if there isn't one.

    Nearest `toe`, not most recent: a daily file is not in broadcast order, and the
    record with the largest `toe` before the epoch can easily be a stale one that
    was superseded.  Unhealthy satellites are excluded outright rather than
    de-weighted -- a satellite flagged unhealthy is not a slightly worse
    measurement, it is one the control segment is telling you not to use.

    Returning None rather than the least-bad record is deliberate.  An ephemeris
    stretched hours past its fit interval yields a position that looks entirely
    plausible and is kilometres wrong.
    """
    best, best_age = None, float("inf")
    for record in records:
        if require_healthy and record.sv_health_flag != 0:
            continue
        age = abs((record.toe - tow_s) + (record.week_num - week) * 604800.0)
        if age < best_age:
            best, best_age = record, age
    if best is None or best_age > max_age_s:
        return None
    return best


def ephemerides_for(
    sat_ids,
    tow_s: float,
    week: int,
    epoch: datetime,
    *,
    resources_dir: str | Path | None = None,
    records: dict[int, list[RINEX_LNAVEphemeris]] | None = None,
    verbose: bool = True,
) -> dict[str, LnavEphemeris]:
    """
    One `LnavEphemeris` per satellite, ready for `utils.navigation`.

    Satellites with no usable record are omitted rather than carried with a bad
    ephemeris; the caller sees a shorter dictionary and can say so.
    """
    if records is None:
        records = load_brdc_span(epoch - timedelta(hours=2), epoch + timedelta(hours=2), resources_dir)

    out: dict[str, LnavEphemeris] = {}
    for sat_id in sat_ids:
        prn = int(str(sat_id).lstrip("G"))
        available = records.get(prn)
        if not available:
            if verbose:
                print(f"  {sat_id}: no broadcast ephemeris in the file")
            continue
        record = select_ephemeris(available, tow_s, week)
        if record is None:
            if verbose:
                print(f"  {sat_id}: no healthy ephemeris within "
                      f"{MAX_AGE_S / 3600:.0f} h of the epoch")
            continue
        out[str(sat_id)] = from_rinex(record, str(sat_id))
    return out


GPS_EPOCH = datetime(1980, 1, 6)


def gps_week_and_tow(epoch: datetime) -> tuple[int, float]:
    """
    `(week, time_of_week)` for a datetime, in GPS time.

    GPS time has no leap seconds, so this treats `epoch` as already being GPS time.
    Converting from UTC means adding the leap-second offset first -- 18 s since
    2017 -- which the caller must do, because whether a given datetime is UTC or
    GPS time is not something this function can know.
    """
    delta = epoch - GPS_EPOCH
    week = delta.days // 7
    tow = delta.total_seconds() - week * 604800.0
    return int(week), float(tow)


def datetime_from_gps(week: int, tow_s: float) -> datetime:
    """Inverse of `gps_week_and_tow`."""
    return GPS_EPOCH + timedelta(weeks=week, seconds=tow_s)
