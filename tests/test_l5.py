"""
GPS L5 acquisition and tracking.

L5 is the first signal in the repository whose two components share every chip
chip: I and Q are separated by carrier phase, not by time.  These tests pin down
that the correlator keeps them apart, that the 90 degree relationship between
them falls out on its own, and that acquisition hands the loops a Doppler they
can actually pull in.
"""

from __future__ import annotations

import numpy as np
import pytest

import gnss_tools.signals.gps_l5 as gps_l5

from utils import bpsk_acquisition
from utils.bpsk_correlation import correlate__multicomponent
from utils.signal_interfaces import (
    TRACKING_POLICIES,
    GpsL5,
    build_acquisition_code_params,
    build_signals,
)

from . import synthetic

# 10.23 Mcps needs at least ~2 samples/chip; the correlator steps the chip index
# once per sample, so a lower rate would skip chips outright.
SAMP_RATE = 25_000_000
PRN = 1


@pytest.fixture(scope="module")
def l5_definition():
    return build_signals(GpsL5, prns=[PRN])[f"G{PRN:02d}"]


def _correlate(samples, code_set, code_phase_chips, bins, carr_phase=0.0, doppler=0.0):
    out = np.zeros((len(bins), code_set.num_components), dtype=np.complex64)
    correlate__multicomponent(
        samples, SAMP_RATE, carr_phase, doppler, code_set,
        gps_l5.CODE_RATE, code_phase_chips, np.asarray(bins, dtype=np.float64), out,
    )
    return out


# --------------------------------------------------------------------------
# Signal definition
# --------------------------------------------------------------------------

def test_components_are_colocated_not_interleaved(l5_definition):
    """Unlike L2C, both L5 components occupy every chip."""
    code_set = l5_definition.code_set
    assert code_set.names == ("I", "Q")
    np.testing.assert_array_equal(code_set.chips_per_component_chip, [1, 1])
    np.testing.assert_array_equal(code_set.component_offset_chips, [0, 0])
    np.testing.assert_array_equal(code_set.component_code_lengths, [10230, 10230])


def test_primary_period_is_one_millisecond(l5_definition):
    """
    10230 chips at 10.23 Mcps.  This equals the aligned correlator's existing
    interval granularity, which is what makes the tiered-code layer cheap to add.
    """
    code_set = l5_definition.code_set
    period_ms = code_set.pattern_period_chips / gps_l5.CODE_RATE * 1e3
    assert period_ms == pytest.approx(1.0)
    assert l5_definition.primary_period_ms == 1


def test_codes_are_int8(l5_definition):
    """The gnss_tools L5 getters return float64 0/1, unlike the L1CA/L2C ones."""
    assert l5_definition.code_set.codes_flat.dtype == np.int8
    assert set(np.unique(l5_definition.code_set.codes_flat)) <= {-1, 1}


def test_carrier_loop_runs_on_the_pilot(l5_definition):
    """
    Q drives the carrier loop from the start.  Switching to it later, at overlay
    sync, would step the reference 90 degrees and force the PLL to re-pull.
    Costas stays on until the overlay is stripped -- NH20 flips Q every 1 ms, so
    the pilot is not yet effectively dataless.
    """
    policy = TRACKING_POLICIES[GpsL5.signal_id].discriminator_policy
    assert policy.carrier_component == l5_definition.code_set.index_of("Q")
    assert policy.code_components == (0, 1), "delay discriminator should combine I and Q"
    assert policy.costas is True


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------

def test_aligned_replica_recovers_both_components(l5_definition):
    samples = synthetic.generate_l5_samples(
        prn=PRN, start_sec=0.0, duration_sec=1e-3, samp_rate=SAMP_RATE,
        doppler_hz=0.0, code_phase_ms=0.0, nav_bits=False, overlay=False,
    )
    result = _correlate(samples, l5_definition.code_set, 0.0, [0.0])
    n = len(samples)
    assert abs(result[0, 0]) == pytest.approx(n, rel=1e-3), "I did not correlate"
    assert abs(result[0, 1]) == pytest.approx(n, rel=1e-3), "Q did not correlate"


def test_components_are_in_quadrature(l5_definition):
    """
    The correlator needs no quadrature handling: correlating the complex baseband
    against each real code independently puts I on the real axis and Q on the
    imaginary axis by itself.
    """
    samples = synthetic.generate_l5_samples(
        prn=PRN, start_sec=0.0, duration_sec=1e-3, samp_rate=SAMP_RATE,
        doppler_hz=0.0, code_phase_ms=0.0, nav_bits=False, overlay=False,
    )
    corr_i, corr_q = _correlate(samples, l5_definition.code_set, 0.0, [0.0])[0]

    phase_difference_deg = np.degrees(np.angle(corr_q / corr_i))
    assert phase_difference_deg == pytest.approx(90.0, abs=1.0)


def test_components_do_not_leak_into_each_other(l5_definition):
    """A signal carrying only I must leave the Q accumulator near zero."""
    code_i, _ = synthetic.get_l5_codes(PRN)
    n = int(SAMP_RATE * 1e-3)
    chips = (np.arange(n) * (gps_l5.CODE_RATE / SAMP_RATE)).astype(np.int64)
    i_only = code_i[chips % len(code_i)].astype(np.complex64)

    corr_i, corr_q = _correlate(i_only, l5_definition.code_set, 0.0, [0.0])[0]
    assert abs(corr_i) == pytest.approx(n, rel=1e-3)
    assert abs(corr_q) < 0.05 * n, "Q picked up energy from an I-only signal"


def test_wrong_prn_does_not_correlate():
    samples = synthetic.generate_l5_samples(
        prn=PRN, start_sec=0.0, duration_sec=1e-3, samp_rate=SAMP_RATE,
        doppler_hz=0.0, code_phase_ms=0.0, nav_bits=False, overlay=False,
    )
    other = build_signals(GpsL5, prns=[11])["G11"]
    result = _correlate(samples, other.code_set, 0.0, [0.0])
    n = len(samples)
    assert abs(result[0, 0]) < 0.1 * n
    assert abs(result[0, 1]) < 0.1 * n


def test_overlay_flips_fall_on_interval_boundaries(l5_definition):
    """
    Neuman-Hofman advances once per primary code period, which is exactly 1 ms --
    the correlation interval length.  So a 1 ms accumulation never straddles an
    overlay transition and keeps full magnitude, even though the overlay is
    present.  That is what lets Stage 2 treat NH as if it were data.
    """
    for period in range(4):
        samples = synthetic.generate_l5_samples(
            prn=PRN, start_sec=period * 1e-3, duration_sec=1e-3, samp_rate=SAMP_RATE,
            doppler_hz=0.0, code_phase_ms=0.0, nav_bits=False, overlay=True,
        )
        result = _correlate(samples, l5_definition.code_set, 0.0, [0.0])
        n = len(samples)
        assert abs(result[0, 0]) == pytest.approx(n, rel=1e-3), f"I lost magnitude at period {period}"
        assert abs(result[0, 1]) == pytest.approx(n, rel=1e-3), f"Q lost magnitude at period {period}"


def test_early_late_are_symmetric_when_aligned(l5_definition):
    samples = synthetic.generate_l5_samples(
        prn=PRN, start_sec=0.0, duration_sec=1e-3, samp_rate=SAMP_RATE,
        doppler_hz=0.0, code_phase_ms=0.0, nav_bits=False, overlay=False,
    )
    early, prompt, late = _correlate(
        samples, l5_definition.code_set, 0.0, [0.5, 0.0, -0.5]
    )[:, 1]
    assert abs(prompt) > abs(early) and abs(prompt) > abs(late)
    assert abs(early) == pytest.approx(abs(late), rel=0.05)


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------

def _acquire(true_doppler_hz, code_phase_ms, half_bin, prn=PRN, noise_sigma=2.0, seed=0):
    definitions = build_signals(GpsL5, prns=[prn])
    config = bpsk_acquisition.AcquisitionConfiguration(
        # NH caps coherent integration at one primary code period; blocks are
        # combined non-coherently, so the overlay's sign flips are harmless.
        replica_duration_ms=1,
        num_blocks=10,
        sample_rate=SAMP_RATE,
        min_search_doppler_hz=-5000,
        max_search_doppler_hz=5000,
    )
    samples = synthetic.generate_l5_samples(
        prn=prn, start_sec=0.0, duration_sec=config.acq_total_duration_ms * 1e-3,
        samp_rate=SAMP_RATE, doppler_hz=true_doppler_hz, code_phase_ms=code_phase_ms,
        noise_sigma=noise_sigma, rng=np.random.default_rng(seed),
    )
    results = bpsk_acquisition.run_acquisition(
        sample_block=samples,
        sample_block_uptime_epoch_ms=0.0,
        acq_config=config,
        code_parameters=build_acquisition_code_params(GpsL5, definitions),
        prob_false_alaram=1e-7,
        noise_var_method="abscorrvar",
        half_bin_doppler_search=half_bin,
    )
    return results[f"G{prn:02d}"], config


def test_acquisition_detects_and_locates_the_signal():
    code_phase_ms = 0.31
    result, _ = _acquire(1234.0, code_phase_ms, half_bin=True)
    assert result.signal_detected
    # Code phase resolves to one sample.
    assert result.acq_code_phase_seconds * 1e3 == pytest.approx(
        code_phase_ms, abs=2e-3 / SAMP_RATE * 1e3 + 1e-6
    )


def test_acquisition_rejects_an_absent_satellite():
    """
    A PRN 11 replica must not detect a PRN 1 signal.

    L5 codes cross-correlate at about -29 dB, so this is a statement about
    operating point as much as about the codes: at an unrealistically strong
    signal (noise_sigma ~ 2) the cross-correlation peak does clear the detection
    threshold, which is the familiar cross-correlation false lock rather than a
    defect.  noise_sigma = 8 is a realistic level where the true peak is ~30x the
    mismatched one.
    """
    config = bpsk_acquisition.AcquisitionConfiguration(
        replica_duration_ms=1, num_blocks=10, sample_rate=SAMP_RATE,
        min_search_doppler_hz=-5000, max_search_doppler_hz=5000,
    )
    samples = synthetic.generate_l5_samples(
        prn=PRN, start_sec=0.0, duration_sec=config.acq_total_duration_ms * 1e-3,
        samp_rate=SAMP_RATE, doppler_hz=800.0, code_phase_ms=0.2,
        noise_sigma=8.0, rng=np.random.default_rng(3),
    )

    peaks = {}
    for prn in (PRN, 11):
        definitions = build_signals(GpsL5, prns=[prn])
        peaks[prn] = bpsk_acquisition.run_acquisition(
            sample_block=samples, sample_block_uptime_epoch_ms=0.0, acq_config=config,
            code_parameters=build_acquisition_code_params(GpsL5, definitions),
            prob_false_alaram=1e-7, noise_var_method="abscorrvar",
        )[f"G{prn:02d}"]

    assert peaks[PRN].signal_detected, "the transmitted PRN should acquire"
    assert not peaks[11].signal_detected, "an absent PRN should not acquire"
    assert (
        peaks[PRN].normalized_peak_value > 10 * peaks[11].normalized_peak_value
    ), "matched peak should dominate the cross-correlation peak"


def test_l5_codes_have_low_cross_correlation():
    """Underpins the rejection test above: distinct PRNs must be near-orthogonal."""
    code_i, code_q = synthetic.get_l5_codes(PRN)
    for other_prn in (2, 7, 11, 20):
        other_i, other_q = synthetic.get_l5_codes(other_prn)
        for mine, theirs, label in ((code_i, other_i, "I"), (code_q, other_q, "Q")):
            peak = np.abs(
                np.fft.ifft(np.fft.fft(mine) * np.conj(np.fft.fft(theirs)))
            ).max() / len(mine)
            assert peak < 0.1, f"PRN {PRN} vs {other_prn} {label}: {peak:.3f}"


def test_doppler_bin_width_is_one_kilohertz():
    """
    The constraint that motivates the half-bin search: coherent integration is
    capped at 1 ms, so bins are 1 / T_coherent = 1000 Hz wide and the worst-case
    seeding error is 500 Hz -- double a 1 ms FLL's unambiguous range.
    """
    _, config = _acquire(0.0, 0.0, half_bin=False)
    assert config.fft_resolution == pytest.approx(1000.0)


@pytest.mark.parametrize("true_doppler_hz", [0.0, 250.0, 400.0, 500.0, 700.0, -400.0, -500.0])
def test_half_bin_search_halves_worst_case_doppler_error(true_doppler_hz):
    """
    Also fixes the sign: mixing the samples down by half a bin makes the signal
    appear lower, so the recovered Doppler is the bin centre *plus* the offset.
    Getting this backwards would double the error instead of halving it.
    """
    coarse, _ = _acquire(true_doppler_hz, 0.31, half_bin=False)
    fine, _ = _acquire(true_doppler_hz, 0.31, half_bin=True)
    assert coarse.signal_detected and fine.signal_detected

    # 250 Hz is the FLL's unambiguous limit at 1 ms; the half-bin grid's worst
    # case sits exactly on it.
    assert abs(fine.acq_doppler_hz - true_doppler_hz) <= 250.0 + 1e-6
    assert abs(fine.acq_doppler_hz - true_doppler_hz) <= abs(
        coarse.acq_doppler_hz - true_doppler_hz
    ) + 1e-6


def test_half_bin_offset_is_reported_and_applied():
    fine, config = _acquire(500.0, 0.31, half_bin=True)
    assert fine.doppler_offset_hz == pytest.approx(0.5 * config.fft_resolution)
    bin_centre = config.doppler_search_bins[fine.peak_doppler_bin] * config.fft_resolution
    assert fine.acq_doppler_hz == pytest.approx(bin_centre + fine.doppler_offset_hz)


def test_half_bin_search_is_off_by_default():
    """Existing L1CA/L2C callers must be unaffected."""
    coarse, config = _acquire(400.0, 0.31, half_bin=False)
    assert coarse.doppler_offset_hz == 0.0
    assert coarse.acq_doppler_hz % config.fft_resolution == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Acquisition handing off to tracking
# --------------------------------------------------------------------------

def test_acquisition_seeds_tracking_to_convergence():
    """
    End to end on the path the notebook takes: acquire, seed a channel from the
    result, track, and confirm the loops pull in and both components accumulate.
    """
    from utils import sample_streaming, tracking_channel
    from utils.signal_interfaces import create_tracking_channels

    true_doppler_hz, code_phase_ms, prn = 1234.0, 0.31, PRN
    acq_result, _ = _acquire(true_doppler_hz, code_phase_ms, half_bin=True, prn=prn)
    assert acq_result.signal_detected

    definitions = build_signals(GpsL5, prns=[prn])
    loop_params = tracking_channel.TrackingLoopParameters(
        DLL_bandwidth_hz=2.0, PLL_bandwidth_hz=20.0, FLL_bandwidth_hz=50.0,
        nominal_update_period_ms=1, corr_period_ms=1, EPL_chip_spacing=0.5,
    )
    channels = create_tracking_channels(
        GpsL5,
        signals=definitions,
        acquisition_results={f"G{prn:02d}": acq_result},
        tracking_signal_ids=[f"G{prn:02d}"],
        loop_params=loop_params,
        output_capacity=400,
    )
    adapter = channels[f"G{prn:02d}"]

    buffer_ms, num_buffers = 50, 4
    rng = np.random.default_rng(9)
    for i in range(num_buffers):
        samples = synthetic.generate_l5_samples(
            prn=prn, start_sec=i * buffer_ms * 1e-3, duration_sec=buffer_ms * 1e-3,
            samp_rate=SAMP_RATE, doppler_hz=true_doppler_hz, code_phase_ms=code_phase_ms,
            noise_sigma=2.0, rng=rng,
        )
        adapter.process_sample_buffer(
            sample_streaming.SampleBuffer(
                samples=samples, start_uptime_ms=i * buffer_ms, samp_rate=SAMP_RATE
            )
        )

    outputs = adapter.outputs
    count = outputs.output_index
    assert count > 100, "too few epochs to judge convergence"

    final_doppler_error = outputs.doppler_freq_hz[count - 1] - true_doppler_hz
    assert abs(final_doppler_error) < 5.0, f"Doppler did not converge: {final_doppler_error:+.2f} Hz"
    assert adapter.channel.loop_state.mode is tracking_channel.TrackingLoopMode.PLL

    # I and Q are equal power, so while both are integrated over the same 1 ms
    # they should be comparable.  Measured before overlay sync, since afterwards
    # Q integrates over 20 ms while I is capped at 10 ms by its CNAV symbols --
    # see test_data_component_cannot_be_integrated_past_its_symbol.
    early_epochs = slice(20, 60)
    prompt_i = np.abs(outputs.prompt_corr[early_epochs, 0]).mean()
    prompt_q = np.abs(outputs.prompt_corr[early_epochs, 1]).mean()
    assert prompt_i > 0 and prompt_q > 0
    assert 0.5 < prompt_i / prompt_q < 2.0, f"I/Q imbalance: {prompt_i:.0f} vs {prompt_q:.0f}"


# --------------------------------------------------------------------------
# Overlay wipe-off and extended coherent integration
# --------------------------------------------------------------------------

def _track_until_synced(noise_sigma=3.0, buffers=14, buffer_ms=50, seed=4):
    """Acquire, then track long enough for the overlay to lock."""
    from utils import sample_streaming, tracking_channel
    from utils.signal_interfaces import create_tracking_channels

    true_doppler_hz, code_phase_ms = 1500.0, 0.31
    definitions = build_signals(GpsL5, prns=[PRN])
    config = bpsk_acquisition.AcquisitionConfiguration(
        replica_duration_ms=1, num_blocks=20, sample_rate=SAMP_RATE,
        min_search_doppler_hz=-5000, max_search_doppler_hz=5000,
    )
    rng = np.random.default_rng(seed)
    acq = bpsk_acquisition.run_acquisition(
        sample_block=synthetic.generate_l5_samples(
            prn=PRN, start_sec=0.0, duration_sec=config.acq_total_duration_ms * 1e-3,
            samp_rate=SAMP_RATE, doppler_hz=true_doppler_hz,
            code_phase_ms=code_phase_ms, noise_sigma=noise_sigma, rng=rng),
        sample_block_uptime_epoch_ms=0.0, acq_config=config,
        code_parameters=build_acquisition_code_params(GpsL5, definitions),
        prob_false_alaram=1e-7, noise_var_method="abscorrvar",
        half_bin_doppler_search=True,
    )[f"G{PRN:02d}"]
    assert acq.signal_detected

    loop_params = tracking_channel.TrackingLoopParameters(
        DLL_bandwidth_hz=2.0, PLL_bandwidth_hz=20.0, FLL_bandwidth_hz=50.0,
        nominal_update_period_ms=1, corr_period_ms=1, EPL_chip_spacing=0.5,
    )
    adapter = create_tracking_channels(
        GpsL5,
        signals=definitions, acquisition_results={f"G{PRN:02d}": acq},
        tracking_signal_ids=[f"G{PRN:02d}"], loop_params=loop_params,
        output_capacity=4000,
    )[f"G{PRN:02d}"]

    synced_at_epoch = None
    for i in range(buffers):
        was_synced = adapter.channel.overlay_sync.synced
        adapter.process_sample_buffer(sample_streaming.SampleBuffer(
            samples=synthetic.generate_l5_samples(
                prn=PRN, start_sec=i * buffer_ms * 1e-3, duration_sec=buffer_ms * 1e-3,
                samp_rate=SAMP_RATE, doppler_hz=true_doppler_hz,
                code_phase_ms=code_phase_ms, noise_sigma=noise_sigma, rng=rng),
            start_uptime_ms=i * buffer_ms, samp_rate=SAMP_RATE))
        if not was_synced and adapter.channel.overlay_sync.synced and synced_at_epoch is None:
            synced_at_epoch = adapter.outputs.output_index
    return adapter, synced_at_epoch, true_doppler_hz


def test_overlay_syncs_and_switches_to_pilot_tracking():
    adapter, synced_at, _ = _track_until_synced()
    channel = adapter.channel

    assert channel.overlay_sync.synced, "overlay never locked"
    assert synced_at is not None
    assert channel.overlay_sync.confidence > 2.0

    # Everything that must move together at the transition.
    assert channel.coherent_periods == 20, "coherent accumulation did not extend"
    assert channel.policy.costas is False, "pilot should drop Costas wrapping"
    assert channel.policy.code_components == (1,), "delay discriminator should be Q only"
    assert channel.loop_params.nominal_update_period_ms == 20, "loop filter not retuned"


def test_wipe_off_gives_n_fold_coherent_gain():
    """
    The point of the whole layer.  Folding 20 primary periods coherently must
    multiply the prompt by ~20, not by sqrt(20) -- the latter is what you would
    get if the overlay signs were wrong and the folds added incoherently.
    """
    adapter, synced_at, _ = _track_until_synced(noise_sigma=0.0)
    outputs = adapter.outputs
    count = outputs.output_index

    # Stay clear of the transition on both sides.
    before = np.abs(outputs.prompt_corr[synced_at - 40 : synced_at - 2, 1]).mean()
    after = np.abs(outputs.prompt_corr[count - 15 : count, 1]).mean()

    gain = after / before
    assert gain == pytest.approx(20.0, rel=0.05), f"expected ~20x, got {gain:.1f}x"
    assert gain > 2 * np.sqrt(20), "gain is incoherent -- overlay signs are wrong"


def test_coherent_gain_holds_across_noise_levels():
    for noise_sigma in (0.0, 2.0, 4.0):
        adapter, synced_at, _ = _track_until_synced(noise_sigma=noise_sigma)
        outputs = adapter.outputs
        count = outputs.output_index
        before = np.abs(outputs.prompt_corr[synced_at - 40 : synced_at - 2, 1]).mean()
        after = np.abs(outputs.prompt_corr[count - 15 : count, 1]).mean()
        assert after / before == pytest.approx(20.0, rel=0.1), (
            f"sigma={noise_sigma}: gain {after/before:.1f}x"
        )


def test_tracking_still_converges_after_the_switch():
    adapter, _, true_doppler_hz = _track_until_synced()
    outputs = adapter.outputs
    count = outputs.output_index
    final_error = outputs.doppler_freq_hz[count - 1] - true_doppler_hz
    assert abs(final_error) < 5.0, f"Doppler diverged after switch: {final_error:+.2f} Hz"


def test_extending_integration_narrows_the_loops():
    """
    A discrete loop needs Bn*T well under ~0.25.  Extending the epoch 20x
    multiplies T by 20, so the bandwidths must come down or the loop turns
    under-damped and passes measurement noise into the estimates.
    """
    adapter, _, _ = _track_until_synced()
    params = adapter.channel.loop_params
    update_period_sec = params.nominal_update_period_ms * 1e-3
    for bandwidth_hz in (
        params.PLL_bandwidth_hz, params.DLL_bandwidth_hz, params.FLL_bandwidth_hz
    ):
        assert bandwidth_hz * update_period_sec <= 0.1 + 1e-9


def test_data_component_cannot_be_integrated_past_its_symbol():
    """
    Why the post-sync delay discriminator drops to Q alone: L5I carries 100 sps
    CNAV symbols, so a 20 ms epoch spans two of them and they partly cancel.  Q,
    being a true pilot once NH is stripped, keeps growing.
    """
    adapter, _, _ = _track_until_synced(noise_sigma=0.0)
    outputs = adapter.outputs
    count = outputs.output_index
    tail = slice(count - 15, count)
    prompt_i = np.abs(outputs.prompt_corr[tail, 0]).mean()
    prompt_q = np.abs(outputs.prompt_corr[tail, 1]).mean()
    assert prompt_q > 2 * prompt_i, (
        f"expected the pilot to dominate over 20 ms, got I={prompt_i:.0f} Q={prompt_q:.0f}"
    )
