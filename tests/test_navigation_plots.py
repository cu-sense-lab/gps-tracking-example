"""
Smoke tests for the navigation plots.

These do not check that a plot looks right -- nothing automated can.  What they
check is that each helper survives the inputs it will actually meet in notebook
02, which for a navigation solution means a lot of NaN: epochs with too few
satellites, satellites below the horizon, channels that dropped lock.  A plotting
helper that raises on the first NaN fails at the end of a long notebook, which is
the most expensive place to find out.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import plotting  # noqa: E402


@pytest.fixture
def fig():
    figure = plt.figure(figsize=(8, 5))
    yield figure
    plt.close(figure)


SAT_IDS = ["G03", "G07", "G19", "G26"]


def sky_tracks(num_points: int = 20):
    rng = np.random.default_rng(0)
    azimuth, elevation = {}, {}
    for i, sat_id in enumerate(SAT_IDS):
        azimuth[sat_id] = np.linspace(i * 80.0, i * 80.0 + 5.0, num_points)
        elevation[sat_id] = np.linspace(15.0 + i * 15.0, 18.0 + i * 15.0, num_points)
    return azimuth, elevation


# ---------------------------------------------------------------------------
# Skyplot
# ---------------------------------------------------------------------------


def test_skyplot_draws_tracks(fig):
    azimuth, elevation = sky_tracks()
    ax = plotting.plot_skyplot(fig, azimuth, elevation)
    assert ax.name == "polar"
    # Zenith at the centre: the radial axis runs 90 inward to 0 outward.
    assert ax.get_ylim() == (90, 0)


def test_skyplot_accepts_scalar_positions(fig):
    azimuth = {s: 45.0 * i for i, s in enumerate(SAT_IDS)}
    elevation = {s: 30.0 for s in SAT_IDS}
    plotting.plot_skyplot(fig, azimuth, elevation)


def test_skyplot_distinguishes_tracked_from_merely_visible(fig):
    azimuth, elevation = sky_tracks()
    ax = plotting.plot_skyplot(fig, azimuth, elevation, tracked=SAT_IDS[:2])
    labelled = [t.get_text() for t in ax.texts]
    assert set(SAT_IDS).issubset(set(labelled)), "every visible satellite gets a label"


def test_skyplot_colours_by_cn0_and_adds_a_colourbar(fig):
    azimuth, elevation = sky_tracks()
    cn0 = {s: 38.0 + 3 * i for i, s in enumerate(SAT_IDS)}
    plotting.plot_skyplot(fig, azimuth, elevation, cn0_dbhz=cn0, tracked=SAT_IDS)
    assert any(a.get_label() == "<colorbar>" for a in fig.axes)


def test_skyplot_skips_satellites_below_the_horizon(fig):
    azimuth = {"G01": np.array([10.0, 20.0]), "G02": np.array([100.0, 110.0])}
    elevation = {"G01": np.array([-5.0, -2.0]), "G02": np.array([40.0, 41.0])}
    ax = plotting.plot_skyplot(fig, azimuth, elevation)
    assert [t.get_text() for t in ax.texts] == ["G02"]


def test_skyplot_handles_missing_cn0_for_some_satellites(fig):
    azimuth, elevation = sky_tracks()
    cn0 = {SAT_IDS[0]: 42.0, SAT_IDS[1]: np.nan}
    plotting.plot_skyplot(fig, azimuth, elevation, cn0_dbhz=cn0, tracked=SAT_IDS)


def test_skyplot_with_no_visible_satellites_does_not_raise(fig):
    plotting.plot_skyplot(fig, {"G01": 10.0}, {"G01": -30.0})


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


def enu_series(n: int = 100, *, nan_rows: int = 0):
    rng = np.random.default_rng(1)
    enu = np.column_stack([
        rng.normal(0.0, 3.0, n),
        rng.normal(0.0, 4.0, n),
        rng.normal(0.0, 8.0, n),
    ])
    if nan_rows:
        enu[:nan_rows] = np.nan
    return enu


def test_position_plot_draws_scatter_and_time_series(fig):
    ax = plotting.plot_position_enu(fig, enu_series(), np.arange(100) * 0.1)
    assert ax.get_xlabel() == "East [m]"
    assert len(fig.axes) == 2


def test_position_plot_tolerates_epochs_without_a_fix(fig):
    """
    The realistic case.  With five satellites, one dropping out leaves an epoch
    with no fix at all, and those rows come through as NaN.
    """
    plotting.plot_position_enu(fig, enu_series(nan_rows=20), np.arange(100) * 0.1)


def test_position_plot_with_no_fixes_at_all(fig):
    plotting.plot_position_enu(fig, np.full((10, 3), np.nan))


def test_position_plot_with_too_few_points_for_an_ellipse(fig):
    """Two fixes cannot define a covariance; the helper must still draw."""
    plotting.plot_position_enu(fig, enu_series(n=2))


def test_position_plot_without_a_time_axis(fig):
    plotting.plot_position_enu(fig, enu_series())


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


def test_clock_plot_reports_drift(fig):
    t = np.arange(200) * 0.1
    # 2 ppm of drift on top of a 1 km offset.
    bias = 1000.0 + 2e-6 * 2.998e8 * t + np.random.default_rng(2).normal(0, 0.5, len(t))
    ax = plotting.plot_clock_solution(fig, t, bias)
    annotations = " ".join(a.get_text() for a in ax.texts)
    assert "ppm" in annotations
    assert "+2.0" in annotations or "+1.9" in annotations or "+2.1" in annotations


def test_clock_plot_tolerates_gaps(fig):
    t = np.arange(50) * 0.1
    bias = 1000.0 + 5.0 * t
    bias[10:20] = np.nan
    plotting.plot_clock_solution(fig, t, bias)


def test_clock_plot_with_a_single_fix(fig):
    """One point cannot support a line fit, and must not raise."""
    plotting.plot_clock_solution(fig, np.array([0.0]), np.array([1000.0]))


def test_clock_plot_with_no_fixes(fig):
    plotting.plot_clock_solution(fig, np.arange(5.0), np.full(5, np.nan))


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


def test_residual_plot_reports_rms_in_the_title(fig):
    rng = np.random.default_rng(3)
    residuals = rng.normal(0.0, 2.0, (50, len(SAT_IDS)))
    ax = plotting.plot_pseudorange_residuals(fig, np.arange(50) * 0.1, residuals, SAT_IDS)
    assert "RMS" in ax.get_title()


def test_residual_plot_tolerates_a_satellite_with_no_data(fig):
    residuals = np.random.default_rng(4).normal(0.0, 2.0, (50, len(SAT_IDS)))
    residuals[:, 1] = np.nan
    plotting.plot_pseudorange_residuals(fig, np.arange(50) * 0.1, residuals, SAT_IDS)


def test_residual_plot_with_nothing_at_all(fig):
    ax = plotting.plot_pseudorange_residuals(
        fig, np.arange(5.0), np.full((5, 2), np.nan), ["G01", "G02"]
    )
    assert "RMS" not in ax.get_title()


# ---------------------------------------------------------------------------
# Correction magnitudes
# ---------------------------------------------------------------------------


def test_correction_plot_orders_by_magnitude(fig):
    terms = {
        "satellite_clock": np.full((10, 4), 30000.0),
        "troposphere": np.full((10, 4), -3.0),
        "ionosphere": np.full((10, 4), -8.0),
    }
    ax = plotting.plot_correction_magnitudes(fig, terms, SAT_IDS)
    # barh draws bottom-up, so the largest ends up last in tick order.
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[-1] == "satellite clock"
    assert ax.get_xscale() == "log"


def test_correction_plot_ignores_geometry_entries(fig):
    """`correct_pseudoranges` returns elevation and azimuth alongside the range
    corrections; they are not corrections and must not be plotted as metres."""
    terms = {
        "satellite_clock": np.full((5, 2), 1000.0),
        "elevation_deg": np.full((5, 2), 45.0),
        "azimuth_deg": np.full((5, 2), 180.0),
    }
    ax = plotting.plot_correction_magnitudes(fig, terms, ["G01", "G02"])
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["satellite clock"]


def test_correction_plot_with_no_terms(fig):
    plotting.plot_correction_magnitudes(fig, {}, [])


def test_correction_plot_ignores_all_nan_terms(fig):
    terms = {"satellite_clock": np.full((5, 2), np.nan), "troposphere": np.full((5, 2), 2.0)}
    ax = plotting.plot_correction_magnitudes(fig, terms, ["G01", "G02"])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["troposphere"]
