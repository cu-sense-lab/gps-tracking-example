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
        self.corr_grid = np.zeros(
            (config.num_delays, config.num_dopplers, num_components), dtype=np.complex64
        )
        # Every delay/doppler bin accumulates the same number of samples per call,
        # so a single running count is sufficient.
        self.corr_count = 0

    def reset(self) -> None:
        self.corr_grid.fill(0.0)
        self.corr_count = 0


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


@dataclass
class TrackingLoopParameters:
    DLL_bandwidth_hz: float
    PLL_bandwidth_hz: float
    FLL_bandwidth_hz: float
    nominal_update_period_ms: float
    corr_period_ms: int
    EPL_chip_spacing: float = 0.5
    prompt_corr_circ_length_threshold: float = 0.9

    def __post_init__(self):
        update_period_seconds = self.nominal_update_period_ms * 1e-3
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

    def __init__(self, capacity: int, num_components: int = 1):
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
        self.output_index = 0


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
        self.outputs = SignalTrackingOutputs(
            capacity=output_capacity, num_components=num_components
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
            duration_ms=self.loop_params.corr_period_ms,
        )

        # Weights for non-coherent code combining, aligned to policy.code_components.
        self._code_weights = signal_params.code_set.power_weights[
            list(self.policy.code_components)
        ]

        # Hidden option/flag variables
        self._ignore_loop_updates = False


    def _combine_code_magnitude(self, corr_vector: np.ndarray) -> float:
        """
        Power-weighted non-coherent magnitude across the delay-discriminator components.

        For a single unit-weight component this is exactly abs(corr), so
        single-component signals are unaffected by the generalisation.
        """
        selected = corr_vector[list(self.policy.code_components)]
        return float(np.sqrt(np.sum(self._code_weights * np.abs(selected) ** 2)))


    def run_loop_filter(self) -> None:
        # Update signal state to reflect parameters at current correlation epoch
        uptime_epoch_ms = self.corr_interval.start_code_phase_ms
        dt_sec = (uptime_epoch_ms - self.signal_state.uptime_epoch_ms) * 1e-3

        # Grid is (delay, doppler, component); assume a single doppler for now.
        # Keep the full component vectors for output, and drive the loops with the
        # components the policy selects.
        early_all, prompt_all, late_all = self.correlator.corr_grid[:, 0]
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
                _wrap_cycles(np.angle(prompt / last_prompt) / (2.0 * np.pi), half_range) / dt_sec
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

        carrier_phase_cycles = (
            self.signal_state.carrier_phase_cycles
            + doppler_freq_hz * dt_sec
            + filt_carr_phase_error_cycles
        )

        code_phase_ms = (
            self.signal_state.code_phase_ms
            + code_rate_ms_per_sec * dt_sec
            + filt_code_phase_error_chips
            / self.signal_params.nominal_code_rate_chips_per_sec
            * 1e3
        )

        self.signal_state.uptime_epoch_ms = uptime_epoch_ms
        self.signal_state.carrier_phase_cycles = carrier_phase_cycles
        self.signal_state.carrier_rate_cyc_per_sec = doppler_freq_hz
        self.signal_state.code_phase_ms = code_phase_ms
        # Must be written back: the correlator and the interval-bound calculation
        # both propagate using signal_state.code_rate_ms_per_sec.
        self.signal_state.code_rate_ms_per_sec = code_rate_ms_per_sec

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
            self.outputs.code_phase_ms[idx] = self.signal_state.code_phase_ms
            self.outputs.delta_omega[idx] = delta_omega
            self.outputs.prompt_corr_circ_length[idx] = circ_length
            self.outputs.output_index += 1


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
                    self.run_loop_filter()

                self.corr_interval.increment()
                self.correlator.reset()
                self.correlator_status = CorrelatorStatus.CLEARED
            else:
                self.correlator_status = CorrelatorStatus.PARTIAL
                break
