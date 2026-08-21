"""
One closed-loop tracking channel, driven by per-signal configuration.

This replaces the previous pair of near-identical channels (one for single-code
BPSK, one for L2C's interleaved CM/CL).  They had drifted apart in their
discriminators and state updates, which is how several bugs went unnoticed; the
code topology now lives in `utils.code_components` and which component drives which
loop lives in `LoopDiscriminatorPolicy`, so signals differ only by data.

Architecture, unchanged from before:
  1. `AlignedCorrelator` accumulates over a code-phase-aligned interval.
  2. The channel consumes only COMPLETE intervals.
  3. Discriminators and loop filters update carrier/code state to the value it
     takes at the start of the next correlation epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.constants import speed_of_light

from utils import sample_streaming

from . import secondary_code
from .bpsk_correlation import correlate__multicomponent
from .code_components import CodeSet, epl_delay_bins


@dataclass
class TrackingSignalState:
    uptime_epoch_ms: float
    code_phase_ms: float
    code_rate_ms_per_sec: float
    carrier_phase_cycles: float
    carrier_rate_cyc_per_sec: float

    def propagate_phase(self, new_uptime_epoch_ms: float) -> tuple[float, float]:
        "Return code phase (ms) and carrier phase (cycles) at new uptime epoch, without modifying state."
        dt_sec = (new_uptime_epoch_ms - self.uptime_epoch_ms) * 1e-3
        new_code_phase_ms = self.code_phase_ms + dt_sec * self.code_rate_ms_per_sec
        new_carrier_phase_cycles = self.carrier_phase_cycles + dt_sec * self.carrier_rate_cyc_per_sec
        return new_code_phase_ms, new_carrier_phase_cycles

    def propagate_to_uptime_ms(self, new_uptime_epoch_ms: float) -> "TrackingSignalState":
        new_code_phase_ms, new_carrier_phase_cycles = self.propagate_phase(new_uptime_epoch_ms)
        return TrackingSignalState(
            uptime_epoch_ms=new_uptime_epoch_ms,
            code_phase_ms=new_code_phase_ms,
            code_rate_ms_per_sec=self.code_rate_ms_per_sec,
            carrier_phase_cycles=new_carrier_phase_cycles,
            carrier_rate_cyc_per_sec=self.carrier_rate_cyc_per_sec,
        )


@dataclass(frozen=True)
class TrackingSignalParameters:
    """Static description of the signal being tracked."""

    code_set: CodeSet
    nominal_code_rate_chips_per_sec: float
    carrier_freq_hz: float
    # Duration of one full pass through the primary code.  Currently only used to
    # document the signal; tiered-code integration will accumulate in units of it.
    primary_period_ms: int = 1

    @property
    def num_components(self) -> int:
        return self.code_set.num_components

    @property
    def chip_period_sec(self) -> float:
        return 1.0 / self.nominal_code_rate_chips_per_sec

    @property
    def chip_length_m(self) -> float:
        return speed_of_light * self.chip_period_sec


@dataclass(frozen=True)
class LoopDiscriminatorPolicy:
    """
    Which correlator components drive which loop.

    `carrier_component` selects the single component the phase/frequency
    discriminators read.  `code_components` are combined non-coherently (and
    power-weighted) for the delay discriminator; with one component of unit weight
    that reduces exactly to the classic single-component form.

    `costas` false means the component is a dataless pilot, so the phase
    discriminator can use the full four-quadrant angle instead of wrapping at
    +/-1/4 cycle -- worth ~6 dB, but only valid once any tiered code has been
    stripped, since an overlay flips sign exactly like data.
    """

    carrier_component: int = 0
    code_components: tuple[int, ...] = (0,)
    costas: bool = True

    def __post_init__(self) -> None:
        if not self.code_components:
            raise ValueError("code_components must not be empty")


@dataclass
class DelayDopplerCorrelatorConfig:
    """
    Correlator bin layout.

    Delay bins are an explicit tuple of chip offsets rather than a count plus a
    step: BOC signals need asymmetric or wider layouts (very-early/very-late) to
    resolve side-peak ambiguity.
    """

    bin_offsets_chips: np.ndarray
    num_dopplers: int = 1
    doppler_offset_hz: float = 0.0
    doppler_step_hz: float = 0.0

    def __post_init__(self) -> None:
        self.bin_offsets_chips = np.ascontiguousarray(self.bin_offsets_chips, dtype=np.float64)

    @property
    def num_delays(self) -> int:
        return len(self.bin_offsets_chips)


class AlignedCorrelator:
    """
    Request-driven correlator engine.

    The tracking channel owns all dynamic signal and epoch state.  The correlator
    owns only static configuration and performs in-place accumulation for the
    epoch that the channel requests.
    """

    def __init__(self, config: DelayDopplerCorrelatorConfig, num_components: int):
        self.config = config
        shape = (config.num_delays, config.num_dopplers, num_components)
        # `corr_grid` holds one correlation interval -- for a signal with a tiered
        # code, one primary code period.  `epoch_grid` is where those intervals are
        # folded, sign-corrected, to build a longer coherent accumulation.  With no
        # tiered code the fold is a plain copy of a single interval, so the two are
        # equivalent and the epoch is one interval long.
        self.corr_grid = np.zeros(shape, dtype=np.complex64)
        self.epoch_grid = np.zeros(shape, dtype=np.complex64)
        # Every delay/doppler bin accumulates the same number of samples per call,
        # so a single running count is sufficient.
        self.corr_count = 0
        self.epoch_count = 0

    def reset(self) -> None:
        self.corr_grid.fill(0.0)
        self.corr_count = 0

    def reset_epoch(self) -> None:
        self.epoch_grid.fill(0.0)
        self.epoch_count = 0

    def fold(self, signs: np.ndarray) -> None:
        """
        Add the finished interval into the epoch accumulator, wiping off the
        tiered code by multiplying each component by its overlay chip.

        Signs are +/-1 int8, so complex64 stays complex64 and a sign of +1 is
        exact -- an un-synced or overlay-free signal folds bit-identically to a
        plain accumulation.

        Must only ever be called on a COMPLETE interval; folding a partially
        accumulated one would corrupt the epoch with a short correlation.
        """
        self.epoch_grid += self.corr_grid * signs[None, None, :]
        self.epoch_count += self.corr_count

    def accumulate(
        self,
        buffer: sample_streaming.SampleBuffer,
        accum_start_uptime_ms: float,
        accum_stop_uptime_ms: float,
        signal_params: TrackingSignalParameters,
        signal_state: TrackingSignalState,
    ) -> None:
        if not (accum_start_uptime_ms < accum_stop_uptime_ms):
            raise ValueError("accumulation stop must be after accumulation start")

        # Determine start and end samples, trimmed to the buffer.
        # Bounds have to be both floor or ceiling; ceiling gives the exact half-open
        # window [start, stop), so consecutive intervals partition the stream.
        accum_start_sample_index = max(
            0,
            int(np.ceil((accum_start_uptime_ms - buffer.start_uptime_ms) * buffer.samp_rate / 1000)),
        )
        accum_stop_sample_index = min(
            len(buffer.samples),
            int(np.ceil((accum_stop_uptime_ms - buffer.start_uptime_ms) * buffer.samp_rate / 1000)),
        )
        if accum_start_sample_index >= accum_stop_sample_index:
            # The window does not overlap this buffer: it can begin inside the final
            # sample period, or be shorter than one sample period. The channel will mark
            # the interval PARTIAL and resume from the interval start on the next buffer.
            return

        actual_accum_start_uptime_ms = (
            buffer.start_uptime_ms + accum_start_sample_index / buffer.samp_rate * 1000
        )
        samples = buffer.samples[accum_start_sample_index:accum_stop_sample_index]
        num_accum_samples = len(samples)

        # propagation uses correlation doppler because we may be accumulating a partial interval
        # (in that case, we want to use the same doppler for propagation as we do for correlation)
        # note: code rate will be decoupled from doppler
        dt_sec = (actual_accum_start_uptime_ms - signal_state.uptime_epoch_ms) * 1e-3
        code_phase_ms = signal_state.code_phase_ms + dt_sec * signal_state.code_rate_ms_per_sec
        code_phase_chips = code_phase_ms * 1e-3 * signal_params.nominal_code_rate_chips_per_sec
        code_rate_chips_per_sec = (
            signal_state.code_rate_ms_per_sec * 1e-3 * signal_params.nominal_code_rate_chips_per_sec
        )
        for i_dopp in range(self.config.num_dopplers):
            corr_doppler_offset_hz = (
                self.config.doppler_offset_hz + i_dopp * self.config.doppler_step_hz
            )
            corr_doppler_hz = signal_state.carrier_rate_cyc_per_sec + corr_doppler_offset_hz
            corr_carrier_phase_cycles = signal_state.carrier_phase_cycles + dt_sec * corr_doppler_hz
            correlate__multicomponent(
                samples,
                buffer.samp_rate,
                corr_carrier_phase_cycles,
                corr_doppler_hz,
                signal_params.code_set,
                code_rate_chips_per_sec,
                code_phase_chips,
                self.config.bin_offsets_chips,
                self.corr_grid[:, i_dopp, :],
            )
        self.corr_count += num_accum_samples


class CorrelatorStatus(Enum):
    """
    Correlator result status flag for a given correlation interval.

    Correlation interval can be complete or partial.
    Partial intervals can be at front, end, or both sides of the accumulation window.
    """

    CLEARED = "UNDEFINED"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"


class TrackingLoopMode(Enum):
    FLL = "FLL"
    PLL = "PLL"


@dataclass
class TrackingLoopState:
    mode: TrackingLoopMode
    history_size: int = 10

    def __post_init__(self):
        self.prompt_corr_history = np.zeros(self.history_size, dtype=complex)
        self.history_index = 0
        self.history_filled = False

    def update_history(self, prompt_corr: complex) -> None:
        self.prompt_corr_history[self.history_index] = prompt_corr
        self.history_index += 1
        if self.history_index >= self.history_size:
            self.history_index = 0
            self.history_filled = True

    def get_last_prompt_corr(self) -> complex | None:
        if self.history_index == 0:
            if self.history_filled:
                return self.prompt_corr_history[-1]
            return None
        return self.prompt_corr_history[self.history_index - 1]

    def compute_prompt_corr_history_circ_length(self, costas: bool = False) -> float:
        iq = (
            self.prompt_corr_history
            if self.history_filled
            else self.prompt_corr_history[: self.history_index]
        )
        if len(iq) == 0:
            return 0.0
        angles = np.angle(iq)
        if costas:
            angles *= 2.0
        return float(np.abs(np.mean(np.exp(1j * angles))))


def estimate_cn0_vsm(prompt_power: np.ndarray, integration_time_s: float) -> float:
    """
    Carrier-to-noise density from prompt power, by the variance summing method.

    VSM (Van Dierendonck) separates signal from noise using the second and fourth
    moments of the prompt magnitude, which is all it needs -- no knowledge of the
    data bits, the carrier phase error or the overlay sign, because none of those
    survive `|P|**2`.  That is what makes it usable before the secondary code is
    synchronised, and why its input needs no wipe-off.

        mu2 = E[|P|^2]                    mu4 = E[|P|^4]
        Pd  = sqrt(2*mu2^2 - mu4)         Pn  = mu2 - Pd
        C/N0 = (Pd / Pn) / T

    `integration_time_s` must be the integration time of the samples given, and the
    samples must all share it: C/N0 is a *density*, so the whole result scales with
    this number.  Feeding it epochs of mixed length silently reports the wrong
    answer, which is why the channel collects one correlation interval at a time.

    Returns NaN rather than raising where the estimator degenerates -- the
    discriminant goes negative at low C/N0, and the noise term vanishes at very
    high -- because a NaN leaves a gap in the record while an exception would take
    down a whole tracking run over one bad window.
    """
    if prompt_power.size == 0:
        return float("nan")
    mu2 = float(np.mean(prompt_power))
    mu4 = float(np.mean(prompt_power ** 2))
    discriminant = 2.0 * mu2 ** 2 - mu4
    if discriminant <= 0.0:
        return float("nan")
    signal_power = float(np.sqrt(discriminant))
    noise_power = mu2 - signal_power
    if noise_power <= 0.0:
        return float("nan")
    return float(10.0 * np.log10((signal_power / noise_power) / integration_time_s))


@dataclass
class CN0EstimatorParameters:
    """
    How the C/N0 estimate is windowed.

    `period_ms` samples -- one correlation interval each -- go into every estimate,
    and a new one is emitted every `hop_ms = period_ms * (1 - overlap_fraction)`.
    Longer periods trade time resolution for a steadier number: measured on the
    rooftop collect, the estimate's 1-sigma spread is about 0.6-0.9 dB at 100 ms,
    0.4-0.9 dB at 200 ms and 0.1-0.2 dB at 1000 ms.

    Note the period must be well under the tracked duration or there is nothing to
    plot: at the 1000 ms default, one second of tracking yields exactly one point.
    """

    period_ms: int = 1000
    overlap_fraction: float = 0.5
    enabled: bool = True

    def __post_init__(self) -> None:
        # VSM's fourth moment is the fragile one, and below roughly a hundred
        # samples it stops being trustworthy.  Refusing beats quietly returning a
        # number nobody can tell is wrong.
        if self.period_ms < 100:
            raise ValueError(
                f"period_ms must be at least 100 correlation intervals for the "
                f"fourth-moment estimate to be meaningful, got {self.period_ms}"
            )
        if not 0.0 <= self.overlap_fraction < 1.0:
            raise ValueError(
                f"overlap_fraction must be in [0, 1), got {self.overlap_fraction}"
            )
        self.hop_ms = int(round(self.period_ms * (1.0 - self.overlap_fraction)))
        if self.hop_ms < 1:
            raise ValueError(
                f"period_ms {self.period_ms} at overlap {self.overlap_fraction} gives a "
                f"hop of {self.hop_ms} intervals; it must be at least 1"
            )


# One correlation interval, in milliseconds.
#
# !!! DO NOT CHANGE THIS. !!!
#
# It is not a tuning knob, and nothing in the repository is correct at any other
# value.  Three separate things depend on it being 1:
#
#   1. It is the granularity at which a sign can be applied.  An overlay chip
#      lasts one primary code period -- 1 ms for L1 C/A, L2C and L5 alike -- so a
#      longer interval would span an overlay flip and cancel itself.
#   2. Interval boundaries land on integer milliseconds of code phase, and every
#      period that matters (code, data symbol, overlay chip) is a whole number of
#      milliseconds.  That is what guarantees no interval straddles a boundary,
#      and it only holds while this divides all of them.  At 3 ms, an L2C interval
#      [18, 21) would cross the CNAV symbol boundary at 20.
#   3. `coherent_duration_ms` is expressed in milliseconds and converted to a
#      count of intervals by dividing by this.
#
# To integrate longer, raise `coherent_duration_ms`.  Intervals are folded --
# with overlay wipe-off between them -- to build the epoch, which is exactly why
# extending integration never means lengthening the interval.
CORRELATION_INTERVAL_MS = 1


@dataclass
class TrackingLoopParameters:
    """
    Loop bandwidths and how long one coherent accumulation lasts.

    `coherent_duration_ms` is the whole story on integration length: it is the
    epoch the discriminators see, the update period the loop gains are built for,
    and (divided by `CORRELATION_INTERVAL_MS`) the number of intervals folded into
    each epoch.  It must divide the shortest interval over which the components
    driving the loops hold a constant sign -- see `validate_coherent_duration`,
    which the channel calls on construction.
    """

    DLL_bandwidth_hz: float
    PLL_bandwidth_hz: float
    FLL_bandwidth_hz: float
    coherent_duration_ms: int = CORRELATION_INTERVAL_MS
    EPL_chip_spacing: float = 0.5
    prompt_corr_circ_length_threshold: float = 0.9

    @property
    def intervals_per_epoch(self) -> int:
        return self.coherent_duration_ms // CORRELATION_INTERVAL_MS

    def __post_init__(self):
        if (
            self.coherent_duration_ms < CORRELATION_INTERVAL_MS
            or self.coherent_duration_ms % CORRELATION_INTERVAL_MS
        ):
            raise ValueError(
                f"coherent_duration_ms must be a positive multiple of "
                f"{CORRELATION_INTERVAL_MS} ms, got {self.coherent_duration_ms}"
            )
        update_period_seconds = self.coherent_duration_ms * 1e-3
        # 1st-order DLL
        # Gain = 4 * Bn * T, where Bn is the DLL bandwidth in Hz and T is the update period in seconds
        self.DLL_filter_coeff = 4.0 * update_period_seconds * self.DLL_bandwidth_hz

        # 2nd-order PLL
        # Gain = 2 * zeta * omega_n * T
        zeta = 1.0 / np.sqrt(2.0)
        omega_n = 2.0 * self.PLL_bandwidth_hz / (zeta + 1.0 / (4.0 * zeta))
        self.PLL_filter_coeffs = (
            2.0 * zeta * omega_n * update_period_seconds
            - 1.5 * omega_n**2 * update_period_seconds**2,
            omega_n**2 * update_period_seconds,
        )

        # 1st-order FLL
        # Gain = 4 * Bn * T, where Bn is the FLL bandwidth in Hz and T is the update period in seconds
        self.FLL_filter_coeff = 4.0 * self.FLL_bandwidth_hz * update_period_seconds


class SignalTrackingOutputs:
    """
    Per-epoch tracking history.

    Correlator outputs are always (capacity, num_components), even for
    single-component signals.  Assigning a whole component vector per epoch is what
    makes it impossible to silently broadcast one component across all columns.
    """

    def __init__(self, capacity: int, num_components: int = 1, cn0_capacity: int = 0):
        self.capacity = capacity
        self.num_components = num_components
        self.uptime_epoch_ms = np.zeros(capacity, dtype=float)
        self.carr_phase_errors_cycles = np.zeros(capacity, dtype=float)
        self.code_phase_errors_chips = np.zeros(capacity, dtype=float)
        self.early_corr = np.zeros((capacity, num_components), dtype=complex)
        self.prompt_corr = np.zeros((capacity, num_components), dtype=complex)
        self.late_corr = np.zeros((capacity, num_components), dtype=complex)
        self.carr_phase_cycles = np.zeros(capacity, dtype=float)
        self.doppler_freq_hz = np.zeros(capacity, dtype=float)
        self.code_phase_ms = np.zeros(capacity, dtype=float)
        self.delta_omega = np.zeros(capacity, dtype=float)
        self.prompt_corr_circ_length = np.zeros(capacity, dtype=float)
        # Which carrier loop actually filtered this epoch.  Recorded because the
        # FLL->PLL handover is invisible in the correlator outputs alone, and
        # reading a lock transient correctly means knowing which loop was running.
        self.pll_mode = np.zeros(capacity, dtype=bool)
        self.output_index = 0

        # C/N0 lands on its own cadence -- one estimate per hop of correlation
        # intervals, not one per epoch -- so it gets its own arrays and its own
        # index rather than being forced onto the epoch axis.
        self.cn0_capacity = cn0_capacity
        self.cn0_dbhz = np.zeros((cn0_capacity, num_components), dtype=float)
        self.cn0_uptime_ms = np.zeros(cn0_capacity, dtype=float)
        self.cn0_index = 0

    @property
    def valid(self) -> slice:
        """
        Slice covering only the epochs actually written. The arrays above are
        pre-allocated to `capacity`; a run shorter than that leaves the tail
        zero-filled, which silently draws a spurious line/point back through
        (0, 0) if plotted unsliced.
        """
        return slice(0, self.output_index)

    @property
    def cn0_valid(self) -> slice:
        """Slice covering only the C/N0 estimates actually written."""
        return slice(0, self.cn0_index)


@dataclass
class CorrelationInterval:
    start_code_phase_ms: int
    duration_ms: int

    @property
    def stop_code_phase_ms(self) -> int:
        return self.start_code_phase_ms + self.duration_ms

    def increment(self) -> None:
        self.start_code_phase_ms += self.duration_ms

    def compute_start_and_stop_uptime_ms(
        self, signal_state: TrackingSignalState
    ) -> tuple[float, float]:
        code_phase_ms = signal_state.code_phase_ms
        code_rate_ms_per_sec = signal_state.code_rate_ms_per_sec
        start_uptime_ms = (
            self.start_code_phase_ms - code_phase_ms
        ) / code_rate_ms_per_sec * 1e3 + signal_state.uptime_epoch_ms
        stop_uptime_ms = (
            self.stop_code_phase_ms - code_phase_ms
        ) / code_rate_ms_per_sec * 1e3 + signal_state.uptime_epoch_ms
        return start_uptime_ms, stop_uptime_ms


def _wrap_cycles(cycles: float, half_range: float) -> float:
    """Wrap a phase in cycles to +/-half_range."""
    period = 2.0 * half_range
    return float(np.mod(cycles + half_range, period) - half_range)


# A discrete loop stays well damped only while its noise bandwidth times its
# update period stays small; the usual guidance is Bn*T <~ 0.25, with 0.1 leaving
# comfortable margin.  Extending coherent integration multiplies T directly, so
# this is the binding constraint on how far integration can be pushed.
MAX_BANDWIDTH_TIME_PRODUCT = 0.1


def validate_coherent_duration(
    signal_params: TrackingSignalParameters,
    policy: LoopDiscriminatorPolicy,
    coherent_duration_ms: int,
    overlay_stripped: bool,
) -> None:
    """
    Reject a coherent integration that would span a sign change.

    An accumulation is only coherent while the thing being accumulated keeps one
    sign.  Two things flip it, and which one binds depends on the state:

      un-stripped overlay   one chip per primary code period.  Until the tiered
                            code is synchronised and wiped off, nothing longer
                            than one period is safe.
      data symbol           `symbol_period_ms` on the component.  A dataless
                            pilot has none, so only Doppler bounds it.

    Every component is checked, not only the ones driving the loops.  One epoch
    serves the whole signal -- each component accumulates over it and each is
    emitted -- so a length that suits the pilot but overruns a data component
    leaves that component's output quietly self-cancelling.  L5 is the case in
    point: its Q pilot is dataless and would tolerate any length, but I carries
    10 ms CNAV symbols, so 10 ms is the signal's limit and 20 ms would wreck I
    while looking perfectly healthy on Q.

    The requirement is divisibility, not merely "fits".  Epochs are anchored to
    multiples of their own length in code phase, so with N dividing the symbol
    period S the epochs tile each symbol exactly; with N = 8 and S = 20 the epoch
    [16, 24) straddles the boundary at 20 however it is aligned.
    """
    for component in signal_params.code_set.components:
        if component.overlay is not None and not overlay_stripped:
            limit_ms = signal_params.primary_period_ms
            reason = f"{component.name!r}'s overlay chip (not yet stripped)"
        elif component.symbol_period_ms is not None:
            limit_ms = component.symbol_period_ms
            reason = f"{component.name!r}'s data symbol"
        else:
            continue  # dataless pilot with no overlay in the way

        if coherent_duration_ms > limit_ms:
            raise ValueError(
                f"coherent_duration_ms {coherent_duration_ms} exceeds {limit_ms} ms, "
                f"the period of {reason}; the accumulation would span a sign change"
            )
        if limit_ms % coherent_duration_ms:
            raise ValueError(
                f"coherent_duration_ms {coherent_duration_ms} does not divide {limit_ms} ms, "
                f"the period of {reason}; epochs are anchored to multiples of their own "
                f"length, so some epoch would straddle a boundary"
            )


def _retune_for_update_period(
    loop_params: TrackingLoopParameters,
    update_period_ms: float,
    max_bandwidth_time_product: float = MAX_BANDWIDTH_TIME_PRODUCT,
) -> TrackingLoopParameters:
    """
    Rebuild the loop filter for a longer update period, narrowing bandwidths as
    needed to keep it stable.

    Two separate things have to happen when coherent integration is extended:

    1. The gains must be recomputed, because all three are proportional to the
       update period.  A 20 ms epoch driving gains built for 1 ms is a 20x
       mistuning.
    2. The bandwidths must be capped.  Holding a 20 Hz PLL across a 20 ms epoch
       gives Bn*T = 0.4, well past the guideline, and the loop becomes
       under-damped: it passes measurement noise straight into the estimates.
       Measured on L5 at 20 ms, Doppler jitter grew from 0.022 to 0.102 Hz as
       noise rose, while the capped loop held flat near 0.047 Hz.  (Coherent gain
       itself is unaffected -- the loss is in the loop, not the accumulation.)

    Narrowing is affordable precisely because the integration is longer: the
    extra coherent gain is what pays for the reduced bandwidth.  Each loop is
    capped independently, so a loop already slow enough is left alone.
    """
    update_period_sec = update_period_ms * 1e-3
    bandwidth_cap_hz = max_bandwidth_time_product / update_period_sec
    return TrackingLoopParameters(
        DLL_bandwidth_hz=min(loop_params.DLL_bandwidth_hz, bandwidth_cap_hz),
        PLL_bandwidth_hz=min(loop_params.PLL_bandwidth_hz, bandwidth_cap_hz),
        FLL_bandwidth_hz=min(loop_params.FLL_bandwidth_hz, bandwidth_cap_hz),
        coherent_duration_ms=int(update_period_ms),
        EPL_chip_spacing=loop_params.EPL_chip_spacing,
        prompt_corr_circ_length_threshold=loop_params.prompt_corr_circ_length_threshold,
    )


class TrackingChannel:
    """
    Tracking channel architecture:
    1. AlignedCorrelator accumulates and emits code-aligned correlations.
    2. TrackingChannel consumes COMPLETE results only.
    3. Loop discriminators + filters update carrier/code to reflect state at start of correlation epochs.
    """

    def __init__(
        self,
        loop_params: TrackingLoopParameters,
        signal_params: TrackingSignalParameters,
        initial_signal_state: TrackingSignalState,
        output_capacity: int = 60000,
        discriminator_policy: LoopDiscriminatorPolicy | None = None,
        correlator_config: DelayDopplerCorrelatorConfig | None = None,
        synced_policy: LoopDiscriminatorPolicy | None = None,
        synced_coherent_duration_ms: int = CORRELATION_INTERVAL_MS,
        initial_overlay_counter: int | None = None,
        cn0_params: CN0EstimatorParameters | None = None,
    ) -> None:
        self.loop_params = loop_params
        self.signal_params = signal_params
        self.policy = discriminator_policy or LoopDiscriminatorPolicy()

        num_components = signal_params.num_components
        for index in (self.policy.carrier_component, *self.policy.code_components):
            if not (0 <= index < num_components):
                raise ValueError(
                    f"discriminator policy references component {index}, but the signal "
                    f"has {num_components} ({signal_params.code_set.names})"
                )

        self.signal_state = initial_signal_state
        self.loop_state = TrackingLoopState(mode=TrackingLoopMode.FLL)
        # --- C/N0 estimator ---------------------------------------------------
        # Sized from the epoch budget: output_capacity epochs of the *initial*
        # coherent duration is the run length in ms, and a channel that later
        # extends covers the same wall time in fewer epochs, so this stays an
        # over-estimate rather than truncating the record.
        self.cn0_params = cn0_params if cn0_params is not None else CN0EstimatorParameters()
        run_duration_ms = output_capacity * loop_params.coherent_duration_ms
        cn0_capacity = (
            run_duration_ms // self.cn0_params.hop_ms + 2 if self.cn0_params.enabled else 0
        )
        # One correlation interval per slot; filled before the epoch machinery
        # touches anything, so the samples are always CORRELATION_INTERVAL_MS long
        # whatever the coherent duration happens to be.
        self._cn0_power = np.zeros(
            (self.cn0_params.period_ms, num_components), dtype=float
        )
        self._cn0_fill = 0          # how many slots hold real data yet
        self._cn0_write = 0         # ring write position
        self._cn0_since_estimate = 0

        self.outputs = SignalTrackingOutputs(
            capacity=output_capacity,
            num_components=num_components,
            cn0_capacity=cn0_capacity,
        )

        if correlator_config is None:
            correlator_config = DelayDopplerCorrelatorConfig(
                bin_offsets_chips=np.array(epl_delay_bins(loop_params.EPL_chip_spacing)),
            )
        self.correlator = AlignedCorrelator(correlator_config, num_components=num_components)
        self.correlator_status = CorrelatorStatus.CLEARED

        start_code_phase_ms, _ = self.signal_state.propagate_phase(
            self.signal_state.uptime_epoch_ms
        )
        self.corr_interval = CorrelationInterval(
            start_code_phase_ms=int(np.ceil(start_code_phase_ms)),
            duration_ms=CORRELATION_INTERVAL_MS,
        )

        # Carry the state from the acquisition epoch onto that first boundary, so
        # `signal_state` is the prior AT AN EPOCH START from the very first epoch.
        # Acquisition reports a code phase mid-period, so without this the first
        # epoch would be the one exception to that invariant.
        first_epoch_uptime_ms, _ = self.corr_interval.compute_start_and_stop_uptime_ms(
            self.signal_state
        )
        self.signal_state = self.signal_state.propagate_to_uptime_ms(first_epoch_uptime_ms)

        # Weights for non-coherent code combining, aligned to policy.code_components.
        self._set_policy(self.policy)

        # --- tiered (overlay) code state -------------------------------------
        # Until the overlay is synced the channel folds one interval per epoch
        # with unit signs, which is arithmetically identical to having no overlay
        # at all -- so signals without one are completely unaffected.
        self._overlays = tuple(c.overlay for c in signal_params.code_set.components)
        self.overlay_sync = secondary_code.build_synchroniser(
            self._overlays, reference_index=self.policy.carrier_component
        )
        self.coherent_duration_ms = loop_params.coherent_duration_ms
        self._epoch_interval_count = 0

        # --- epoch grid anchoring --------------------------------------------
        # Epochs open on a data symbol boundary, so an epoch never straddles one
        # and epoch zero coincides with symbol zero -- which is what later makes
        # the symbols demodulable without a separate alignment search.
        #
        # The period is the shortest symbol among *all* components, not the one
        # driving the loops: on L5 the carrier loop runs on the dataless Q pilot,
        # which has no symbol of its own, while I's 10 ms CNAV symbol is what the
        # shared epoch has to respect.  A signal that is pilot-only everywhere has
        # no boundary to find, and falls back to anchoring on the epoch length.
        symbol_periods = [
            c.symbol_period_ms
            for c in signal_params.code_set.components
            if c.symbol_period_ms is not None
        ]
        self._epoch_anchor_period_ms: int | None = min(symbol_periods) if symbol_periods else None
        # Cleared whenever the epoch length changes, so the new grid re-anchors.
        self._epoch_grid_anchored = False
        self._unit_signs = np.ones(num_components, dtype=np.int8)
        self._synced_policy = synced_policy
        self._synced_coherent_duration_ms = synced_coherent_duration_ms
        # Loop gains scale with the update period, so extending the coherent
        # accumulation demands a retuned filter.  Built up front to keep the
        # transition free of allocation and surprises.
        self._synced_loop_params = (
            _retune_for_update_period(loop_params, synced_coherent_duration_ms)
            if synced_coherent_duration_ms > loop_params.coherent_duration_ms
            else loop_params
        )

        # (validation follows)
        # Both configurations are checked up front, so a bad choice fails at
        # construction rather than part-way through a run.
        validate_coherent_duration(
            signal_params, self.policy, loop_params.coherent_duration_ms,
            overlay_stripped=initial_overlay_counter is not None,
        )
        if synced_coherent_duration_ms > loop_params.coherent_duration_ms:
            validate_coherent_duration(
                signal_params, synced_policy or self.policy,
                synced_coherent_duration_ms, overlay_stripped=True,
            )

        # An overlay phase recovered by acquisition skips the post-lock search
        # entirely: wipe-off and the pilot discriminator are available from the
        # first interval.  Coherent integration still waits for PLL lock.
        if initial_overlay_counter is not None:
            if self.overlay_sync is None:
                raise ValueError(
                    "initial_overlay_counter given for a signal with no tiered code "
                    f"({signal_params.code_set.names})"
                )
            self.overlay_sync.counter = (
                int(initial_overlay_counter) % self.overlay_sync.period
            )
            self.overlay_sync.status = secondary_code.OverlaySyncStatus.SYNCED
            self._on_overlay_synced()

        # Hidden option/flag variables
        self._ignore_loop_updates = False

    def _set_policy(self, policy: LoopDiscriminatorPolicy) -> None:
        self.policy = policy
        self._code_weights = self.signal_params.code_set.power_weights[
            list(policy.code_components)
        ]

    def _combine_code_magnitude(self, corr_vector: np.ndarray) -> float:
        """
        Power-weighted non-coherent magnitude across the delay-discriminator components.

        For a single unit-weight component this is exactly abs(corr), so
        single-component signals are unaffected by the generalisation.
        """
        selected = corr_vector[list(self.policy.code_components)]
        return float(np.sqrt(np.sum(self._code_weights * np.abs(selected) ** 2)))

    def _on_overlay_synced(self) -> None:
        """
        Switch to pilot tracking now that the tiered code can be stripped.

        Wipe-off begins immediately -- it is a sign multiply, and once the overlay
        phase is known it is always right.  The discriminator can also drop Costas
        wrapping here: with the overlay removed a pilot really is dataless, so the
        full four-quadrant angle is available and the squaring loss goes away.

        What does NOT happen here is lengthening the coherent accumulation.  That
        is limited by Doppler error rather than by the overlay, so it waits for
        lock -- see `_maybe_extend_coherent_duration`.
        """
        if self._synced_policy is not None:
            self._set_policy(self._synced_policy)

        # Whatever is part-way through the epoch was folded without wipe-off, so
        # mixing it into the first coherent accumulation would partly cancel it.
        # Start the first wiped-off epoch clean.
        self.correlator.reset_epoch()
        self._epoch_interval_count = 0

    def _maybe_extend_coherent_duration(self) -> None:
        """
        Lengthen the coherent accumulation, once the carrier is actually tracked.

        Coherent integration of length T loses `sinc(df * T)` to a Doppler error
        df, so a 20 ms accumulation needs df well inside 25 Hz.  Acquisition seeds
        it to half a Doppler bin -- 50 Hz on L5's 100 Hz grid -- where df*T is 1.0,
        a null.  The FLL is what removes that error, so extension waits on PLL
        lock, not on overlay sync.

        While sync was itself gated on PLL these two could be done together and the
        distinction never showed.  It matters as soon as the overlay phase arrives
        from acquisition instead, because then sync happens at epoch zero with the
        loops still pulling in.

        The loop filter is retuned in the same step: every gain scales with the
        update period, so a 20 ms epoch driving gains built for 1 ms is a 20x
        mistuning.  The two must not be separated.
        """
        if self.coherent_duration_ms == self._synced_coherent_duration_ms:
            return
        if self.overlay_sync is None or not self.overlay_sync.synced:
            return
        if self.loop_state.mode is not TrackingLoopMode.PLL:
            return

        # Switch exactly on a symbol boundary, so the longer grid is anchored the
        # moment it starts.
        #
        # Waiting here rather than switching now and dropping intervals until the
        # boundary is what keeps the transition lossless: the shorter epochs keep
        # producing output right up to the boundary.  Only a grid with no shorter
        # length to fall back on -- one configured multi-interval from the start --
        # has to discard intervals instead (see `_complete_interval`).
        anchor_ms = self._epoch_anchor_period_ms or self._synced_coherent_duration_ms
        next_interval_code_phase_ms = (
            self.corr_interval.start_code_phase_ms + self.corr_interval.duration_ms
        )
        if next_interval_code_phase_ms % anchor_ms != 0:
            return

        self.coherent_duration_ms = self._synced_coherent_duration_ms
        self.loop_params = self._synced_loop_params
        self.correlator.reset_epoch()
        self._epoch_interval_count = 0
        self._epoch_grid_anchored = True

    def run_loop_filter(self) -> None:
        # Update signal state to reflect parameters at current correlation epoch.
        #
        # This is the epoch's real time on the sample stream, not its code phase.
        # The two are different quantities that merely start out numerically close,
        # and taking one for the other was a live defect: the correlator propagates
        # carrier phase as `carrier_phase + dt_sec * doppler`, so a `dt_sec` wrong by
        # the acquired code phase `d` injects `d * doppler` cycles.  A constant would
        # be harmless, but the loop is busy correcting Doppler, so the error moves by
        # `d * delta_doppler` and the FLL -- which divides by dt -- reads it back as
        # an apparent frequency error `d / T` times larger than the correction that
        # caused it.  That is positive feedback with gain `d / CORRELATION_INTERVAL_MS`.
        #
        # Code phase never showed it: the same `d` sits inside `code_phase_ms` and
        # cancels, so replicas stayed chip-aligned while the phase angles went to
        # noise.  L1 C/A and L5 survived because acquisition reports `d < 1 ms` for
        # them, keeping the gain under 1.  L2C acquires on CM's 20 ms period, so `d`
        # runs to 20 ms and real signals diverged above about 4 ms -- roughly 80% of
        # possible code phases.  Every L2C scenario used 0.61 ms, so nothing caught it.
        # `signal_state` is the PRIOR for the epoch that just closed, and it already
        # sits at that epoch's start -- the previous run propagated it there, and the
        # constructor placed the first one.  Nothing has to be propagated before
        # correcting; the corrections apply to it directly.
        uptime_epoch_ms = self.signal_state.uptime_epoch_ms

        # How long this epoch ran, in stream time.
        #
        # Epoch boundaries are defined in CODE PHASE -- that is the alignment
        # requirement -- so finding when one occurs needs the code rate.
        # `compute_start_and_stop_uptime_ms` is the single place that conversion
        # happens; everywhere else a dt is a difference of uptimes, because code phase
        # is an estimated state like carrier phase, not a clock.  `corr_interval` is
        # still the last interval of the epoch -- the caller increments it afterwards
        # -- so its stop is where the next epoch begins.
        #
        # The FLL turns a phase difference into a rate and so needs this before the
        # corrections below exist, hence the prior.  Strictly it wants the PREVIOUS
        # epoch's span, since the two prompts it differences are one epoch apart; the
        # two differ only on the single epoch where `coherent_duration_ms` changes,
        # and that transition is gated on PLL lock, where `delta_omega` is logged but
        # does not drive the loop.
        _, next_epoch_uptime_ms = self.corr_interval.compute_start_and_stop_uptime_ms(
            self.signal_state
        )
        epoch_span_sec = (next_epoch_uptime_ms - uptime_epoch_ms) * 1e-3

        # Grid is (delay, doppler, component); assume a single doppler for now.
        # Keep the full component vectors for output, and drive the loops with the
        # components the policy selects.  The epoch grid is the folded, overlay
        # wiped-off accumulation -- one interval long until the overlay syncs.
        early_all, prompt_all, late_all = self.correlator.epoch_grid[:, 0]
        prompt = prompt_all[self.policy.carrier_component]

        # Compute loop discriminators
        # Costas wrapping (+/-1/4 cycle) makes a 180 degree data or overlay flip
        # invisible; a dataless pilot can use the full +/-1/2 cycle range instead.
        half_range = 0.25 if self.policy.costas else 0.5

        # Phase discriminator for PLL
        delta_theta = _wrap_cycles(np.angle(prompt) / (2.0 * np.pi), half_range)

        # Frequency discriminator for FLL
        last_prompt = self.loop_state.get_last_prompt_corr()
        if last_prompt is None or last_prompt == 0.0:
            delta_omega = 0.0
        else:
            # Wrapped the same way as the phase discriminator so bit flips are not
            # read as frequency error, then divided by the epoch spacing to give Hz.
            delta_omega = (
                _wrap_cycles(np.angle(prompt / last_prompt) / (2.0 * np.pi), half_range)
                / epoch_span_sec
            )

        # Code discriminator for DLL
        EPL_chip_spacing = self.loop_params.EPL_chip_spacing
        early_mag = self._combine_code_magnitude(early_all)
        late_mag = self._combine_code_magnitude(late_all)
        prompt_mag = self._combine_code_magnitude(prompt_all)
        denom = early_mag + late_mag + 2.0 * prompt_mag
        if denom < 1e-12:
            delta_eta = 0.0
        else:
            delta_eta = (2 - EPL_chip_spacing) * (early_mag - late_mag) / denom

        # Apply loop filters to discriminators
        self.loop_state.update_history(prompt)
        circ_length = self.loop_state.compute_prompt_corr_history_circ_length(
            costas=self.policy.costas
        )

        filt_code_phase_error_chips = self.loop_params.DLL_filter_coeff * delta_eta
        # Captured before the branch below, which may switch the mode: the epoch
        # that triggers the handover is still filtered by the FLL, so that is the
        # mode this epoch should be logged under.
        epoch_used_pll = self.loop_state.mode is TrackingLoopMode.PLL
        if self.loop_state.mode == TrackingLoopMode.FLL:
            if (
                self.loop_state.history_filled
                and circ_length > self.loop_params.prompt_corr_circ_length_threshold
            ):
                self.loop_state.mode = TrackingLoopMode.PLL
            filt_carr_phase_error_cycles = 0.0
            filt_doppler_freq_error_hz = self.loop_params.FLL_filter_coeff * delta_omega
        else:
            filt_carr_phase_error_cycles = self.loop_params.PLL_filter_coeffs[0] * delta_theta
            filt_doppler_freq_error_hz = self.loop_params.PLL_filter_coeffs[1] * delta_theta

        if self._ignore_loop_updates:
            filt_carr_phase_error_cycles = 0.0
            filt_doppler_freq_error_hz = 0.0
            filt_code_phase_error_chips = 0.0

        # State update and propagation
        # Rate (Doppler) is updated first, then code/carrier phase is updated and propagated based on updated rate.
        doppler_freq_hz = self.signal_state.carrier_rate_cyc_per_sec + filt_doppler_freq_error_hz
        # Code rate is slaved to carrier doppler, expressed as ms of code phase per second.
        code_rate_ms_per_sec = (1.0 + doppler_freq_hz / self.signal_params.carrier_freq_hz) * 1e3

        # ---- posterior: the prior plus the filtered corrections. -------------
        # Still at this epoch's start, because that is where the prior was.  This is
        # the estimate describing the epoch just measured, and it is what gets logged.
        carrier_phase_cycles = (
            self.signal_state.carrier_phase_cycles + filt_carr_phase_error_cycles
        )
        code_phase_ms = (
            self.signal_state.code_phase_ms
            + filt_code_phase_error_chips
            / self.signal_params.nominal_code_rate_chips_per_sec
            * 1e3
        )

        idx = self.outputs.output_index
        # Outputs beyond capacity are silently dropped (output_index stops advancing).
        if idx < self.outputs.capacity:
            self.outputs.uptime_epoch_ms[idx] = uptime_epoch_ms
            self.outputs.carr_phase_errors_cycles[idx] = delta_theta
            self.outputs.code_phase_errors_chips[idx] = delta_eta
            self.outputs.early_corr[idx] = early_all
            self.outputs.prompt_corr[idx] = prompt_all
            self.outputs.late_corr[idx] = late_all
            self.outputs.carr_phase_cycles[idx] = carrier_phase_cycles
            self.outputs.doppler_freq_hz[idx] = doppler_freq_hz
            self.outputs.code_phase_ms[idx] = code_phase_ms
            self.outputs.delta_omega[idx] = delta_omega
            self.outputs.prompt_corr_circ_length[idx] = circ_length
            self.outputs.pll_mode[idx] = epoch_used_pll
            self.outputs.output_index += 1

        # ---- prior for the next epoch: the posterior, propagated to its start. ----
        # Where the next epoch begins is re-derived from the POSTERIOR, because the
        # corrections just moved the state and the corrected rate is what the NCO runs
        # at across the span.  From there the propagation is purely time-driven:
        # `propagate_to_uptime_ms` advances code phase and carrier phase by
        # rate x dt over an uptime difference.  Neither is asserted to a boundary
        # value -- both are estimates carried forward.
        posterior = TrackingSignalState(
            uptime_epoch_ms=uptime_epoch_ms,
            code_phase_ms=code_phase_ms,
            code_rate_ms_per_sec=code_rate_ms_per_sec,
            carrier_phase_cycles=carrier_phase_cycles,
            carrier_rate_cyc_per_sec=doppler_freq_hz,
        )
        _, next_epoch_uptime_ms = self.corr_interval.compute_start_and_stop_uptime_ms(
            posterior
        )
        self.signal_state = posterior.propagate_to_uptime_ms(next_epoch_uptime_ms)

        # Checked every epoch rather than only on the FLL->PLL edge, so that a
        # channel started directly in PLL, or synced from acquisition after lock,
        # still picks it up.  It is a no-op once applied.
        self._maybe_extend_coherent_duration()

    def _record_interval_for_cn0(self) -> None:
        """
        Feed one correlation interval's prompt power to the C/N0 estimator.

        Called at the very top of `_complete_interval`, which is deliberate on two
        counts.  It runs before `fold()`, so the sample is one raw interval and is
        unaffected by `coherent_duration_ms` -- the estimate's integration time has
        to be the interval, not the epoch.  And it runs before the epoch-anchoring
        return, because an interval dropped while waiting for a symbol boundary is
        a perfectly good correlation; it is discarded for epoch-tiling reasons, and
        excluding it would punch a hole in the C/N0 record at the start of every
        channel.
        """
        params = self.cn0_params
        if not params.enabled or self.outputs.cn0_capacity == 0:
            return

        # Prompt is delay tap 1; every component, since magnitude costs nothing to
        # keep and comparing I against Q is the point.
        prompt = self.correlator.corr_grid[1, 0, :]
        self._cn0_power[self._cn0_write] = np.abs(prompt) ** 2
        self._cn0_write = (self._cn0_write + 1) % params.period_ms
        self._cn0_fill = min(self._cn0_fill + 1, params.period_ms)
        self._cn0_since_estimate += 1

        if self._cn0_fill < params.period_ms or self._cn0_since_estimate < params.hop_ms:
            return
        self._cn0_since_estimate = 0

        idx = self.outputs.cn0_index
        if idx >= self.outputs.cn0_capacity:
            return
        integration_time_s = CORRELATION_INTERVAL_MS * 1e-3
        for component in range(self.outputs.num_components):
            self.outputs.cn0_dbhz[idx, component] = estimate_cn0_vsm(
                self._cn0_power[:, component], integration_time_s
            )
        # Stamped at the end of the window the estimate covers.
        self.outputs.cn0_uptime_ms[idx] = self.signal_state.uptime_epoch_ms
        self.outputs.cn0_index += 1

    def _complete_interval(self) -> None:
        """
        Fold one finished correlation interval into the epoch, and run the loop
        filter once the epoch is full.

        This is the whole of the tiered-code machinery, and it sits *on top of*
        the interval logic rather than inside it: the PARTIAL/COMPLETE handling in
        `process_sample_buffer` is untouched, and only ever hands complete
        intervals here.  A signal with no overlay folds with unit signs and an
        epoch of one interval, which is arithmetically the old behaviour.
        """
        self._record_interval_for_cn0()

        if self.overlay_sync is None:
            signs = self._unit_signs
        else:
            # Feed the synchroniser the raw prompt, before wipe-off: it is looking
            # for the overlay's own sign pattern, which wipe-off would remove.
            # Gated on PLL lock, because an un-locked carrier leaves the prompts
            # rotating and the correlation meaningless.
            if not self.overlay_sync.synced and self.loop_state.mode is TrackingLoopMode.PLL:
                prompt = self.correlator.corr_grid[1, 0, self.policy.carrier_component]
                if self.overlay_sync.observe(complex(prompt)):
                    self._on_overlay_synced()
            # Signs are read *after* a possible sync so that the interval which
            # triggered it is itself wiped off correctly, rather than folded raw
            # into the freshly reset epoch.
            signs = self.overlay_sync.signs(self._overlays)
            self.overlay_sync.advance()

        # Open the epoch grid on a data symbol boundary.
        #
        # `validate_coherent_duration` rejects an epoch length that does not divide
        # every component's symbol period, but divisibility only keeps epochs
        # inside those boundaries if the grid is *anchored*.  The seeded code phase
        # is wherever acquisition landed, so a grid begun there is offset by an
        # arbitrary number of milliseconds: on L5 a 5 ms epoch started at code
        # phase 3 crosses the 10 ms CNAV symbol on every other epoch, and I
        # collapses whenever consecutive symbols differ while the dataless Q looks
        # perfect.
        #
        # Anchoring on the symbol period rather than merely on the epoch length is
        # the stronger choice, and it is deliberate: both avoid straddling, but
        # only this one makes epoch zero coincide with symbol zero, so the symbols
        # can later be read off the epoch index without a separate search.
        #
        # Anchoring happens once per grid -- at construction and again after every
        # change of epoch length -- not at every epoch.  Once the first epoch is on
        # a boundary, the epoch length divides the symbol period and the tiling
        # stays on it.  Intervals before the boundary are dropped: at most
        # `symbol_period - 1` of them, once.  The caller resets the interval grid
        # immediately after this returns, and the epoch accumulator is already
        # empty whenever the interval count is zero.
        #
        # A caveat worth stating: this locates the boundary in *code phase*, so it
        # is a true symbol boundary only for a signal whose acquisition recovers
        # symbol phase -- L5 via the NH20 counter, L2C via CM's 20 ms period.  GPS
        # L1 C/A carries no symbol phase in its 1 ms code, so its grid is anchored
        # deterministically but arbitrarily, and real bit synchronisation remains a
        # separate step this repository does not implement.
        # A single-interval epoch is exempt: it cannot straddle anything, because
        # every period that matters is a whole number of milliseconds and intervals
        # are integer-ms aligned.  Waiting for a boundary would only throw away
        # intervals, and symbol alignment is still recoverable afterwards since
        # every symbol boundary is also an epoch boundary.  So the wait here only
        # ever applies to a channel configured multi-interval from construction:
        # the post-lock extension anchors by waiting at the shorter length instead
        # (see `_maybe_extend_coherent_duration`), which discards nothing.
        if not self._epoch_grid_anchored:
            if self.coherent_duration_ms <= CORRELATION_INTERVAL_MS:
                self._epoch_grid_anchored = True
            else:
                anchor_ms = self._epoch_anchor_period_ms or self.coherent_duration_ms
                if self.corr_interval.start_code_phase_ms % anchor_ms:
                    return
                self._epoch_grid_anchored = True

        self.correlator.fold(signs)
        self._epoch_interval_count += 1

        if self._epoch_interval_count >= self.coherent_duration_ms // CORRELATION_INTERVAL_MS:
            self.run_loop_filter()
            self.correlator.reset_epoch()
            self._epoch_interval_count = 0

    def process_sample_buffer(self, buffer: sample_streaming.SampleBuffer) -> None:
        while True:
            # Determine code period integration bounds for current signal state
            (
                corr_interval_start_uptime_ms,
                corr_interval_stop_uptime_ms,
            ) = self.corr_interval.compute_start_and_stop_uptime_ms(self.signal_state)

            # If previous correlation interval status was partial, check whether this will complete the interval
            # If it will not, then we can ignore the last correlation (reset correlator) and start new accum.
            if (
                self.correlator_status == CorrelatorStatus.PARTIAL
                and buffer.start_uptime_ms > corr_interval_stop_uptime_ms
            ):
                self.corr_interval.increment()
                self.correlator.reset()
                self.correlator_status = CorrelatorStatus.CLEARED
                continue

            # Accumulate (either a new interval or continuation of a partial interval)
            self.correlator.accumulate(
                buffer,
                accum_start_uptime_ms=corr_interval_start_uptime_ms,
                accum_stop_uptime_ms=corr_interval_stop_uptime_ms,
                signal_params=self.signal_params,
                signal_state=self.signal_state,
            )

            # Check if correlation was completed (i.e. if accum. stop time is within current buffer)
            if corr_interval_stop_uptime_ms < buffer.stop_uptime_ms:
                # An interval that fell entirely behind the stream (non-contiguous buffers)
                # accumulates nothing; skip it rather than running the loop filter on zeros.
                if self.correlator.corr_count > 0:
                    self.correlator_status = CorrelatorStatus.COMPLETE
                    self._complete_interval()

                self.corr_interval.increment()
                self.correlator.reset()
                self.correlator_status = CorrelatorStatus.CLEARED
            else:
                self.correlator_status = CorrelatorStatus.PARTIAL
                break
