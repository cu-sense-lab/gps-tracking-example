"""
Coherent integration shorter than the replica.

The replica -- and therefore the FFT -- must span a whole code period, because the
correlation is circular.  How much *data* goes into one coherent integration is a
separate choice, and capping it is the only way to acquire a signal whose data
symbol is no longer than its code period.

GPS L2 CM is that case: its 20 ms code period is exactly its CNAV symbol, so a
full-period coherent integration at an arbitrary alignment straddles a symbol
boundary.  The interesting part is *how* that fails.  At the true Doppler the two
halves subtract, but the search maximises over Doppler, and a mid-window sign flip
has a spectral null at DC with its energy displaced to roughly +/-1/(2T).  So the
peak does not vanish -- it moves, and acquisition reports a confidently detected
signal at a Doppler that is wrong by a bin.  A silently biased tracking seed is a
good deal worse than a missed detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import bpsk_acquisition as bpsk_acq
from utils.signal_interfaces import (
    GpsL2C,
    build_acquisition_code_params,
    build_signals,
)

from . import synthetic

FS = 5_000_000
PRN = 1
# Boundary at the window edge vs dead centre.  _nav_bits flips at every multiple
# of 20 ms, so a dwell starting at 10 ms puts a flip exactly mid-window.
START_NO_FLIP_SEC = 0.000
START_FLIP_MID_SEC = 0.010
CODE_PHASE_MS = 0.61


def _config(**overrides) -> bpsk_acq.AcquisitionConfiguration:
    params = dict(
        replica_duration_ms=20,
        num_blocks=1,
        sample_rate=FS,
        min_search_doppler_hz=-2000,
        max_search_doppler_hz=2000,
    )
    params.update(overrides)
    return bpsk_acq.AcquisitionConfiguration(**params)


# --------------------------------------------------------------------------
# Block packing
# --------------------------------------------------------------------------

def test_packing_reduces_to_reshape_when_coherent_equals_replica():
    """The pre-existing path must be untouched, bit for bit."""
    rng = np.random.default_rng(0)
    length, num_blocks = 64, 3
    samples = (
        rng.normal(size=length * num_blocks) + 1j * rng.normal(size=length * num_blocks)
    ).astype(np.complex64)

    packed = bpsk_acq.pack_coherent_blocks(samples, num_blocks, length, length)
    np.testing.assert_array_equal(packed, samples.reshape(num_blocks, length))


def test_short_blocks_sit_where_they_were_received():
    """
    Block j lands at offset j*coherent within the code period, not at 0.

    This is the whole reason the padding is not just `block[:n_coh]`: block j was
    received j*n_coh samples later, so its correlation peak sits that much further
    along the code.  Padding every block at 0 leaves four peaks at four different
    lags and the square-law sum smears them instead of stacking them.
    """
    length, coherent, num_blocks = 20, 5, 4
    samples = np.arange(1, num_blocks * coherent + 1).astype(np.complex64)

    packed = bpsk_acq.pack_coherent_blocks(samples, num_blocks, length, coherent)

    for j in range(num_blocks):
        expected = np.zeros(length, dtype=np.complex64)
        expected[j * coherent : (j + 1) * coherent] = samples[
            j * coherent : (j + 1) * coherent
        ]
        np.testing.assert_array_equal(packed[j], expected)


def test_blocks_wrap_at_the_code_period_boundary():
    """A block straddling the period wraps; circular correlation handles it."""
    length, coherent, num_blocks = 20, 8, 3
    samples = np.arange(1, num_blocks * coherent + 1).astype(np.complex64)

    packed = bpsk_acq.pack_coherent_blocks(samples, num_blocks, length, coherent)

    # Block 2 starts at (2*8) % 20 = 16 and runs 8 samples: 16..19, then 0..3.
    np.testing.assert_array_equal(packed[2][16:20], samples[16:20])
    np.testing.assert_array_equal(packed[2][0:4], samples[20:24])
    assert np.all(packed[2][4:16] == 0), "the rest of the window must stay zero"


def test_packing_rejects_a_short_sample_block():
    with pytest.raises(ValueError, match="need 20 samples"):
        bpsk_acq.pack_coherent_blocks(np.zeros(10, dtype=np.complex64), 4, 20, 5)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_coherent_defaults_to_the_replica_length():
    config = _config()
    assert config.coherent_duration_ms == 20.0
    assert config.coherent_length_samples == config.replica_length_samples
    assert config.doppler_response_width_hz == pytest.approx(config.fft_resolution)


def test_short_coherent_gives_a_grid_finer_than_the_response():
    """
    Grid spacing follows the FFT length, mainlobe width follows the coherent
    length.  Separating them is what leaves the Doppler grid 4x oversampled, so
    the residual error stays grid-limited rather than mainlobe-limited.
    """
    config = _config(coherent_duration_ms=5.0, num_blocks=4)
    assert config.fft_resolution == pytest.approx(50.0)
    assert config.doppler_response_width_hz == pytest.approx(200.0)
    # Same dwell as one 20 ms block, so the two are directly comparable.
    assert config.acq_total_duration_ms == pytest.approx(20.0)


def test_coherent_longer_than_the_replica_is_rejected():
    with pytest.raises(ValueError, match="coherent <= replica"):
        _config(coherent_duration_ms=25.0)


def test_coherent_shorter_than_a_sample_is_rejected():
    with pytest.raises(ValueError, match="shorter than one sample"):
        _config(coherent_duration_ms=1e-6)


# --------------------------------------------------------------------------
# Acquiring L2 CM across a CNAV symbol boundary
# --------------------------------------------------------------------------

def _acquire(start_sec, noise_sigma, *, coherent=None, num_blocks=1, seed=7):
    config = _config(coherent_duration_ms=coherent, num_blocks=num_blocks)
    samples = synthetic.generate_l2c_samples(
        prn=PRN,
        start_sec=start_sec,
        duration_sec=config.acq_total_duration_ms * 1e-3,
        samp_rate=FS,
        doppler_hz=0.0,
        code_phase_ms=CODE_PHASE_MS,
        nav_bits=True,
        noise_sigma=noise_sigma,
        rng=np.random.default_rng(seed),
    )
    signals = build_signals(GpsL2C, prns=[PRN])
    result = bpsk_acq.run_acquisition(
        sample_block=samples,
        sample_block_uptime_epoch_ms=0.0,
        acq_config=config,
        code_parameters=build_acquisition_code_params(GpsL2C, signals),
        prob_false_alarm_total=1e-6,
        noise_var_method="abscorrvar",
    )[f"G{PRN:02d}"]
    return result, config


def _expected_code_phase_ms(start_sec: float) -> float:
    """Code phase at the dwell start, wrapped into one 20 ms CM period."""
    return (CODE_PHASE_MS + start_sec * 1e3) % 20.0


def test_a_symbol_flip_mid_window_biases_the_doppler_estimate():
    """
    The failure that motivates the whole feature, and it is not a missed
    detection: the peak survives, moves a Doppler bin, and is reported as a
    confident acquisition.
    """
    clean, _ = _acquire(START_NO_FLIP_SEC, 20.0)
    flipped, _ = _acquire(START_FLIP_MID_SEC, 20.0)

    assert clean.signal_detected and flipped.signal_detected

    assert clean.acq_doppler_hz == pytest.approx(0.0, abs=1e-6)
    assert abs(flipped.acq_doppler_hz) >= 50.0, (
        "a mid-window symbol flip should displace the peak off the true Doppler"
    )
    # Same num_blocks, so the normalised peaks are directly comparable.
    assert flipped.normalized_peak_value < 0.7 * clean.normalized_peak_value


def test_short_coherent_blocks_recover_the_doppler_across_a_flip():
    """Four 5 ms blocks over the same 20 ms of data put the Doppler back."""
    flipped, _ = _acquire(START_FLIP_MID_SEC, 20.0)
    split, _ = _acquire(START_FLIP_MID_SEC, 20.0, coherent=5.0, num_blocks=4)

    assert split.signal_detected
    assert split.acq_doppler_hz == pytest.approx(0.0, abs=1e-6)
    assert abs(split.acq_doppler_hz) < abs(flipped.acq_doppler_hz)


def test_short_coherent_blocks_still_resolve_the_code_phase():
    """
    Guards the offset placement end to end: if the blocks were padded at 0 the
    four peaks would land at four different lags and the reported code phase
    would be whichever one noise happened to favour.
    """
    split, _ = _acquire(START_FLIP_MID_SEC, 20.0, coherent=5.0, num_blocks=4)
    expected_ms = _expected_code_phase_ms(START_FLIP_MID_SEC)
    assert split.acq_code_phase_seconds * 1e3 == pytest.approx(expected_ms, abs=1e-3)


def test_splitting_does_not_disturb_a_flip_free_dwell():
    """Shortening coherent integration is safe when there is nothing to avoid."""
    whole, _ = _acquire(START_NO_FLIP_SEC, 20.0)
    split, _ = _acquire(START_NO_FLIP_SEC, 20.0, coherent=5.0, num_blocks=4)

    assert split.signal_detected
    assert split.acq_doppler_hz == pytest.approx(whole.acq_doppler_hz, abs=1e-6)
    assert split.acq_code_phase_seconds == pytest.approx(
        whole.acq_code_phase_seconds, abs=2.0 / FS
    )
