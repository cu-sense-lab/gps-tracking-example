"""
Runs a TrackingScenario through the tracking stack and returns its outputs.

This is the one file that is expected to change when the tracking channels are
unified.  The golden baseline it produces (tests/golden/*.npz) is the invariant:
after the refactor this driver is rewritten against the new API and must still
reproduce the same numbers.

Outputs are returned as a plain dict of arrays so the comparison is independent of
whatever dataclass holds them.
"""

from __future__ import annotations

import numpy as np

from utils import sample_streaming
from utils import tracking_channel
from utils.signal_interfaces import TRACKING_POLICIES, GpsL1CA, GpsL2C, GpsL5, build_signals

from . import synthetic
from .scenarios import TrackingScenario

# Loop settings mirror notebooks/02-acquisition-and-tracking.ipynb.
BLOCK_DURATION_MS = 1
LOOP_KWARGS = dict(
    DLL_bandwidth_hz=2.0,
    PLL_bandwidth_hz=20.0,
    FLL_bandwidth_hz=50.0,
    coherent_duration_ms=BLOCK_DURATION_MS,
    EPL_chip_spacing=0.5,
    prompt_corr_circ_length_threshold=0.9,
)

_SIGNAL_TYPES = {
    "L1CA": GpsL1CA,
    "L2C": GpsL2C,
    "L5": GpsL5,
}

_FAMILY_GENERATORS = {
    "L1CA": synthetic.generate_l1ca_samples,
    "L2C": synthetic.generate_l2c_samples,
    "L5": synthetic.generate_l5_samples,
}

OUTPUT_FIELDS = (
    "uptime_epoch_ms",
    "carr_phase_errors_cycles",
    "code_phase_errors_chips",
    "early_corr",
    "prompt_corr",
    "late_corr",
    "carr_phase_cycles",
    "doppler_freq_hz",
    "code_phase_ms",
    "delta_omega",
    "prompt_corr_circ_length",
)


def _build_channel(scenario: TrackingScenario):
    """
    Build a channel straight from the shipped signal definitions, so the tests
    exercise the same configuration the notebook does rather than a parallel one.
    """
    signal_type = _SIGNAL_TYPES[scenario.family]
    loop_params = tracking_channel.TrackingLoopParameters(**LOOP_KWARGS)
    policy = TRACKING_POLICIES[signal_type.signal_type_id]

    signal = build_signals(signal_type, prns=[scenario.prn])[f"G{scenario.prn:02d}"]

    # Seed the loops the way acquisition would: offset Doppler, offset code phase.
    seed_doppler_hz = scenario.doppler_hz - scenario.doppler_error_hz
    seed_code_phase_ms = (
        scenario.code_phase_ms
        + scenario.code_error_chips * 1e3 / signal_type.tracking_code_rate_chips_per_sec
    )
    initial_state = tracking_channel.TrackingSignalState(
        uptime_epoch_ms=0.0,
        code_phase_ms=seed_code_phase_ms,
        code_rate_ms_per_sec=(1.0 + seed_doppler_hz / signal_type.carrier_freq_hz) * 1e3,
        carrier_phase_cycles=0.0,
        carrier_rate_cyc_per_sec=seed_doppler_hz,
    )

    signal_params = tracking_channel.TrackingSignalParameters(
        code_set=signal.code_set,
        nominal_code_rate_chips_per_sec=signal_type.tracking_code_rate_chips_per_sec,
        carrier_freq_hz=signal_type.carrier_freq_hz,
        primary_period_ms=signal_type.primary_period_ms,
    )
    return tracking_channel.TrackingChannel(
        loop_params=loop_params,
        signal_params=signal_params,
        initial_signal_state=initial_state,
        output_capacity=scenario.duration_ms // BLOCK_DURATION_MS + 16,
        discriminator_policy=policy.discriminator_policy,
        # Must mirror create_tracking_channels, or the goldens would silently
        # cover a different configuration than the notebook runs: overlay
        # wipe-off without the extended integration and policy switch it enables.
        synced_policy=policy.synced_discriminator_policy,
        synced_coherent_duration_ms=policy.synced_coherent_duration_ms,
    )


def _generate_buffer(scenario: TrackingScenario, buffer_index: int, rng) -> np.ndarray:
    generate = _FAMILY_GENERATORS[scenario.family]
    return generate(
        prn=scenario.prn,
        start_sec=buffer_index * scenario.buffer_duration_ms * 1e-3,
        duration_sec=scenario.buffer_duration_ms * 1e-3,
        samp_rate=scenario.samp_rate,
        doppler_hz=scenario.doppler_hz,
        code_phase_ms=scenario.code_phase_ms,
        noise_sigma=scenario.noise_sigma,
        nav_bits=scenario.nav_bits,
        rng=rng,
    )


def run_scenario(scenario: TrackingScenario) -> dict[str, np.ndarray]:
    """Track a synthetic signal end to end; return trimmed output arrays."""
    channel = _build_channel(scenario)
    rng = np.random.default_rng(scenario.rng_seed)

    for buffer_index in range(scenario.duration_ms // scenario.buffer_duration_ms):
        samples = _generate_buffer(scenario, buffer_index, rng)
        channel.process_sample_buffer(
            sample_streaming.SampleBuffer(
                samples=samples,
                start_uptime_ms=buffer_index * scenario.buffer_duration_ms,
                samp_rate=scenario.samp_rate,
            )
        )

    outputs = channel.outputs
    count = outputs.output_index
    result = {name: np.asarray(getattr(outputs, name))[:count] for name in OUTPUT_FIELDS}
    result["output_index"] = np.asarray(count)
    result["final_mode"] = np.asarray(channel.loop_state.mode.name)
    # Which component the loops actually ran on. Not component 0 in general: L5
    # tracks the Q pilot, and its I component is data-limited so its correlators
    # say nothing about lock quality.
    result["carrier_component"] = np.asarray(channel.policy.carrier_component)
    return result
