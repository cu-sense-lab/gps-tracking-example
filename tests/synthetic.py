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
import gnss_tools.signals.gps_l5 as gps_l5


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


def get_l5_codes(prn: int) -> tuple[np.ndarray, np.ndarray]:
    """+/-1 int8 (L5I, L5Q) primary codes."""
    code_i = (1 - 2 * gps_l5.get_GPS_L5I_code_sequence(prn)).astype(np.int8)
    code_q = (1 - 2 * gps_l5.get_GPS_L5Q_code_sequence(prn)).astype(np.int8)
    return code_i, code_q


def get_l5_overlays() -> tuple[np.ndarray, np.ndarray]:
    """+/-1 int8 Neuman-Hofman overlays (NH10 on I, NH20 on Q)."""
    nh_i = (1 - 2 * gps_l5.NEUMAN_HOFFMAN_SEQ_L5I).astype(np.int8)
    nh_q = (1 - 2 * gps_l5.NEUMAN_HOFFMAN_SEQ_L5Q).astype(np.int8)
    return nh_i, nh_q


def _nav_bits(code_period_index: np.ndarray, periods_per_symbol: int) -> np.ndarray:
    """
    Deterministic alternating data modulation, so bit flips are exercised.

    Keyed to the code period, not to absolute time.  The data symbol is synchronous
    with the primary code in every signal here (IS-GPS-200 for L1 C/A and L2 CNAV,
    IS-GPS-705 for L5), so a flip falls exactly on a code period boundary and can
    never land inside a correlation interval.  Keying it to `t` instead put the
    flip `code_phase_ms` into each period -- a fixture artefact that made a
    symbol-length coherent accumulation look worse than it is.
    """
    return 1 - 2 * ((code_period_index // periods_per_symbol) % 2 == 1)


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
    chip_index = chips.astype(np.int64)
    # 20 ms nav bit = 20 code periods of 1 ms.
    data = _nav_bits(chip_index // gps_l1ca.CODE_LENGTH, 20) if nav_bits else 1.0

    samples = code[chip_index % len(code)] * data
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
    # One CNAV symbol is exactly one CM period: 10230 CM chips = 20460 combined.
    data = _nav_bits(k // (2 * gps_l2c.CODE_LENGTH_L2CM), 1) if nav_bits else 1.0

    chips = np.where(k % 2 == 0, cm[(k // 2) % len(cm)] * data, cl[(k // 2) % len(cl)])
    samples = (chips * np.exp(2j * np.pi * doppler_hz * t)).astype(np.complex64)
    return _add_noise(samples, noise_sigma, rng)


def generate_l5_samples(
    *,
    prn: int,
    start_sec: float,
    duration_sec: float,
    samp_rate: float,
    doppler_hz: float,
    code_phase_ms: float,
    noise_sigma: float = 0.0,
    nav_bits: bool = True,
    overlay: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Baseband GPS L5: I and Q in carrier quadrature, both at 10.23 Mcps.

    L5I carries CNAV symbols at 100 sps (10 ms each); L5Q is a dataless pilot.
    Both carry a Neuman-Hofman overlay that flips sign every primary code period
    (1 ms) -- NH10 on I, NH20 on Q -- which is why Q is not yet effectively
    dataless at 1 ms coherent integration.

    Note the chip rate demands a high sample rate: at least ~20 Msps to keep two
    samples per chip.
    """
    code_i, code_q = get_l5_codes(prn)
    nh_i, nh_q = get_l5_overlays()

    n = int(round(samp_rate * duration_sec))
    t = start_sec + np.arange(n) / samp_rate

    code_rate = gps_l5.CODE_RATE * (1.0 + doppler_hz / gps_l5.CARRIER_FREQ)
    chips = code_phase_ms * 1e-3 * gps_l5.CODE_RATE + t * code_rate
    chip_index = chips.astype(np.int64)

    # Overlay chips advance once per primary code period.
    period_index = chip_index // gps_l5.PRIMARY_CODE_LENGTH
    overlay_i = nh_i[period_index % len(nh_i)] if overlay else 1
    overlay_q = nh_q[period_index % len(nh_q)] if overlay else 1

    # One CNAV symbol is 10 ms = 10 primary code periods, which is also NH10's
    # period -- that alignment is what makes NH10 sync deliver symbol sync.
    data = _nav_bits(period_index, 10) if nav_bits else 1

    in_phase = code_i[chip_index % len(code_i)] * overlay_i * data
    quadrature = code_q[chip_index % len(code_q)] * overlay_q

    samples = (in_phase + 1j * quadrature) * np.exp(2j * np.pi * doppler_hz * t)
    return _add_noise(samples.astype(np.complex64), noise_sigma, rng)


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
