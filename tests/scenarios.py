"""
Tracking regression scenarios.

These are pure data: signal family, ground truth, and the seeding error a realistic
acquisition would hand to the tracking loops.  They must stay stable across refactors
so the golden baseline remains comparable.

Seeding errors are expressed in physical units (Hz, chips) rather than internal
state units, so they stay meaningful if the state representation changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingScenario:
    name: str
    family: str  # "L1CA" | "L2C" | "L5"
    prn: int
    duration_ms: int
    samp_rate: float
    # ground truth
    doppler_hz: float
    code_phase_ms: float
    # acquisition seeding error handed to the loops
    doppler_error_hz: float
    code_error_chips: float
    # channel conditions
    noise_sigma: float
    nav_bits: bool
    rng_seed: int = 0
    buffer_duration_ms: int = 200


# Chosen to span the realistic acquisition seeding envelope:
#   L1 C/A acquisition at 4 ms coherent -> 250 Hz bins -> up to +/-125 Hz error
#   code phase to within one sample -> ~0.2 chip at 5 Msps
# plus clean noise-free cases that pin down exact correlator behaviour.
SCENARIOS: tuple[TrackingScenario, ...] = (
    TrackingScenario(
        name="l1ca_clean",
        family="L1CA",
        prn=1,
        duration_ms=2000,
        samp_rate=5e6,
        doppler_hz=1234.0,
        code_phase_ms=0.37,
        doppler_error_hz=0.0,
        code_error_chips=0.0,
        noise_sigma=0.0,
        nav_bits=False,
    ),
    TrackingScenario(
        name="l1ca_navbits_seeded",
        family="L1CA",
        prn=1,
        duration_ms=2000,
        samp_rate=5e6,
        doppler_hz=1234.0,
        code_phase_ms=0.37,
        doppler_error_hz=125.0,
        code_error_chips=0.25,
        noise_sigma=0.0,
        nav_bits=True,
    ),
    TrackingScenario(
        name="l1ca_noisy",
        family="L1CA",
        prn=7,
        duration_ms=2000,
        samp_rate=5e6,
        doppler_hz=-2100.0,
        code_phase_ms=0.81,
        doppler_error_hz=60.0,
        code_error_chips=0.5,
        noise_sigma=3.0,
        nav_bits=True,
        rng_seed=42,
    ),
    TrackingScenario(
        name="l2c_clean",
        family="L2C",
        prn=1,
        duration_ms=2000,
        samp_rate=5e6,
        doppler_hz=-870.0,
        code_phase_ms=0.61,
        doppler_error_hz=0.0,
        code_error_chips=0.0,
        noise_sigma=0.0,
        nav_bits=False,
    ),
    TrackingScenario(
        name="l2c_navbits_noisy",
        family="L2C",
        prn=1,
        duration_ms=2000,
        samp_rate=5e6,
        doppler_hz=-870.0,
        code_phase_ms=0.61,
        doppler_error_hz=100.0,
        code_error_chips=0.25,
        noise_sigma=3.0,
        nav_bits=True,
        rng_seed=1,
    ),
    # L5 runs at 25 Msps (10.23 Mcps needs >=2 samples/chip), so these are kept
    # short: 200 ms is ~5x the samples of a 2 s L1 C/A run.  Doppler seeding
    # errors reflect the half-bin acquisition search, whose worst case is 250 Hz
    # -- the edge of a 1 ms FLL's unambiguous range.
    TrackingScenario(
        name="l5_clean",
        family="L5",
        prn=1,
        duration_ms=200,
        samp_rate=25e6,
        doppler_hz=1500.0,
        code_phase_ms=0.31,
        doppler_error_hz=0.0,
        code_error_chips=0.0,
        noise_sigma=0.0,
        nav_bits=False,
        buffer_duration_ms=50,
    ),
    TrackingScenario(
        name="l5_navbits_seeded",
        family="L5",
        prn=1,
        duration_ms=200,
        samp_rate=25e6,
        doppler_hz=1500.0,
        code_phase_ms=0.31,
        doppler_error_hz=250.0,
        code_error_chips=0.25,
        noise_sigma=0.0,
        nav_bits=True,
        buffer_duration_ms=50,
    ),
    TrackingScenario(
        name="l5_noisy",
        family="L5",
        prn=7,
        duration_ms=200,
        samp_rate=25e6,
        doppler_hz=-2400.0,
        code_phase_ms=0.72,
        doppler_error_hz=125.0,
        code_error_chips=0.5,
        noise_sigma=3.0,
        nav_bits=True,
        rng_seed=5,
        buffer_duration_ms=50,
    ),
)


def get_scenario(name: str) -> TrackingScenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"unknown scenario: {name}")
