"""
Synthetic GNSS signal generators for regression and unit tests.

The repository's sample collects live in `local-data/`, which is gitignored and not
guaranteed to be present.  These generators produce deterministic, noise-controlled
signals so the correlator and tracking loops can be exercised without any data
dependency, and so a golden baseline can be diffed across refactors.

Conventions match the tracking code:
  - Code phase is expressed in *ms of code*; the correlator converts via
    `code_phase_ms * 1e-3 * nominal_code_rate_chips_per_sec`.
  - Code rate is slaved to carrier Doppler by `1 + doppler / carrier_freq`.
  - Samples are complex64 baseband (carrier already mixed to near-DC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import gnss_tools.signals.gps_l1ca as gps_l1ca
import gnss_tools.signals.gps_l2c as gps_l2c


@dataclass(frozen=True)
class SyntheticTruth:
    """Ground-truth parameters used to generate a signal, for assertions."""

    doppler_hz: float
    code_phase_ms: float
    samp_rate: float
    carrier_freq_hz: float
    nominal_code_rate_chips_per_sec: float


def get_l1ca_code(prn: int) -> np.ndarray:
    """+/-1 int8 L1 C/A primary code."""
    return (1 - 2 * gps_l1ca.get_GPS_L1CA_code_sequence(prn)).astype(np.int8)


def get_l2c_codes(prn: int) -> tuple[np.ndarray, np.ndarray]:
    """+/-1 int8 (L2CM, L2CL) primary codes."""
    cm = (1 - 2 * gps_l2c.get_GPS_L2CM_code_sequence(prn)).astype(np.int8)
    cl = (1 - 2 * gps_l2c.get_GPS_L2CL_code_sequence(prn)).astype(np.int8)
    return cm, cl



def _nav_bits(t: np.ndarray, bit_period_sec: float) -> np.ndarray:
    """Deterministic alternating data modulation, so bit flips are exercised."""
    return 1 - 2 * (((t // bit_period_sec).astype(np.int64)) % 2 == 1)


def generate_l1ca_samples(
    *,
    prn: int,
    start_sec: float,
    duration_sec: float,
    samp_rate: float,
    doppler_hz: float,
    code_phase_ms: float,
    noise_sigma: float = 0.0,
    nav_bits: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Baseband GPS L1 C/A: a single BPSK code with optional 50 bps data."""
    code = get_l1ca_code(prn)
    n = int(round(samp_rate * duration_sec))
    t = start_sec + np.arange(n) / samp_rate

    code_rate = gps_l1ca.CODE_RATE * (1.0 + doppler_hz / gps_l1ca.CARRIER_FREQ)
    chips = code_phase_ms * 1e-3 * gps_l1ca.CODE_RATE + t * code_rate
    data = _nav_bits(t, 0.020) if nav_bits else 1.0

    samples = code[chips.astype(np.int64) % len(code)] * data
    samples = (samples * np.exp(2j * np.pi * doppler_hz * t)).astype(np.complex64)
    return _add_noise(samples, noise_sigma, rng)


def generate_l2c_samples(
    *,
    prn: int,
    start_sec: float,
    duration_sec: float,
    samp_rate: float,
    doppler_hz: float,
    code_phase_ms: float,
    noise_sigma: float = 0.0,
    nav_bits: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Baseband GPS L2C: CM and CL chip-interleaved on a 1.023 Mcps combined stream.

    Even combined chips carry CM (with CNAV data), odd ones carry the
    dataless CL pilot.  This is the topology the tracking correlator must invert.
    """
    cm, cl = get_l2c_codes(prn)
    n = int(round(samp_rate * duration_sec))
    t = start_sec + np.arange(n) / samp_rate

    combined_rate = gps_l2c.CODE_RATE_L2CLM * (1.0 + doppler_hz / gps_l2c.CARRIER_FREQ)
    k = (code_phase_ms * 1e-3 * gps_l2c.CODE_RATE_L2CLM + t * combined_rate).astype(np.int64)
    data = _nav_bits(t, 0.020) if nav_bits else 1.0

    chips = np.where(k % 2 == 0, cm[(k // 2) % len(cm)] * data, cl[(k // 2) % len(cl)])
    samples = (chips * np.exp(2j * np.pi * doppler_hz * t)).astype(np.complex64)
    return _add_noise(samples, noise_sigma, rng)



def _add_noise(
    samples: np.ndarray, sigma: float, rng: np.random.Generator | None
) -> np.ndarray:
    if sigma <= 0.0:
        return samples
    if rng is None:
        raise ValueError("rng is required when noise_sigma > 0 (tests must be deterministic)")
    n = len(samples)
    noise = rng.normal(0.0, sigma, n) + 1j * rng.normal(0.0, sigma, n)
    return (samples + noise).astype(np.complex64)
