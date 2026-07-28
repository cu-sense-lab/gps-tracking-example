from dataclasses import dataclass
from enum import Enum

import numpy as np

from utils import sample_streaming

from scipy.constants import speed_of_light

from .bpsk_correlation import correlate__delay


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
    


@dataclass
class TrackingSignalParameters:
    code_seq: np.ndarray
    nominal_code_rate_chips_per_sec: float
    carrier_freq_hz: float

    @property
    def code_length_chips(self) -> int:
        return len(self.code_seq)
    
    @property
    def chip_period_sec(self) -> float:
        return 1.0 / self.nominal_code_rate_chips_per_sec
    
    @property
    def chip_length_m(self) -> float:
        return speed_of_light * self.chip_period_sec


@dataclass
class DelayDopplerCorrelatorConfig:
    num_delays: int
    delay_offset_chips: float
    delay_step_chips: float
    num_dopplers: int
    doppler_offset_hz: float
    doppler_step_hz: float


class AlignedCorrelator:
    """
    Request-driven correlator engine.

    The tracking channel owns all dynamic signal and epoch state. The correlator
    owns only static configuration and performs in-place accumulation for the
    epoch that the channel requests.
    """

    def __init__(
        self,
        config: DelayDopplerCorrelatorConfig,
    ):
        self.config = config
        self.corr_grid = np.zeros((config.num_delays, config.num_dopplers), dtype=np.complex64)
        self.corr_counts = np.zeros((config.num_delays, config.num_dopplers), dtype=int)

    def reset(self) -> None:
        self.corr_grid.fill(0.0)
        self.corr_counts.fill(0)

    def accumulate(
        self,
        buffer: sample_streaming.SampleBuffer,
        accum_start_uptime_ms: float,
        accum_stop_uptime_ms: float,
        signal_params: TrackingSignalParameters,
        signal_state: TrackingSignalState,
    ) -> None:
        
        # first, need to determine start and end samples
        accum_start_sample_index = int((accum_start_uptime_ms - buffer.start_uptime_ms) * buffer.samp_rate / 1000)
        accum_stop_sample_index = int((accum_stop_uptime_ms - buffer.start_uptime_ms) * buffer.samp_rate / 1000)
        if not (accum_start_sample_index < accum_stop_sample_index):
            raise ValueError("accumulation stop must be after accumulation start")

        accum_start_sample_index = max(0, accum_start_sample_index)
        accum_stop_sample_index = min(len(buffer.samples), accum_stop_sample_index)
        
        actual_accum_start_uptime_ms = buffer.start_uptime_ms + accum_start_sample_index / buffer.samp_rate * 1000
        samples = buffer.samples[accum_start_sample_index:accum_stop_sample_index]
        num_accum_samples = len(samples)
        if num_accum_samples == 0:
            # NOTE: this should not happen
            print(f"Warning: no samples to accumulate for interval {accum_start_uptime_ms} ms to {accum_stop_uptime_ms} ms (buffer from {buffer.start_uptime_ms} ms to {buffer.stop_uptime_ms} ms)")
            print(f"accum_start_sample_index: {accum_start_sample_index}, accum_stop_sample_index: {accum_stop_sample_index}")
            raise ValueError("no samples to accumulate")

        # propagation uses correlation doppler because we may be accumulating a partial interval
        # (in that case, we want to use the same doppler for propagation as we do for correlation)
        # note: code rate will be decoupled from doppler
        dt_sec = (actual_accum_start_uptime_ms - signal_state.uptime_epoch_ms) * 1e-3
        code_phase_ms = signal_state.code_phase_ms + dt_sec * signal_state.code_rate_ms_per_sec
        code_phase_chips = code_phase_ms * 1e-3 * signal_params.nominal_code_rate_chips_per_sec
        code_rate_chips_per_sec = signal_state.code_rate_ms_per_sec * 1e-3 * signal_params.nominal_code_rate_chips_per_sec
        for i_dopp in range(self.config.num_dopplers):
            corr_doppler_offset_hz = self.config.doppler_offset_hz + i_dopp * self.config.doppler_step_hz
            corr_doppler_hz = signal_state.carrier_rate_cyc_per_sec + corr_doppler_offset_hz
            corr_carrier_phase_cycles = signal_state.carrier_phase_cycles + dt_sec * corr_doppler_hz
            correlate__delay(
                samples,
                buffer.samp_rate,
                corr_carrier_phase_cycles,
                corr_doppler_hz,
                signal_params.code_seq,
                signal_params.code_length_chips,
                code_rate_chips_per_sec,
                code_phase_chips,
                self.config.num_delays,
                self.config.delay_offset_chips,
                self.config.delay_step_chips,
                self.corr_grid[:, i_dopp],
            )
            self.corr_counts[:, i_dopp] += num_accum_samples


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
        self.DLL_filter_coeff = 4.0 * update_period_seconds * self.DLL_bandwidth_hz

        xi = 1.0 / np.sqrt(2.0)
        omega_n = self.PLL_bandwidth_hz / 0.53
        self.PLL_filter_coeffs = (
            2.0 * xi * omega_n * update_period_seconds
            - 1.5 * omega_n**2 * update_period_seconds**2,
            omega_n**2 * update_period_seconds,
        )
        self.FLL_filter_coeff = 4.0 * self.FLL_bandwidth_hz * update_period_seconds


@dataclass
class SignalTrackingOutputs:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.uptime_epoch_ms = np.zeros(capacity, dtype=float)
        self.carr_phase_errors_cycles = np.zeros(capacity, dtype=float)
        self.code_phase_errors_chips = np.zeros(capacity, dtype=float)
        self.early_corr = np.zeros(capacity, dtype=complex)
        self.prompt_corr = np.zeros(capacity, dtype=complex)
        self.late_corr = np.zeros(capacity, dtype=complex)
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

    def compute_start_and_stop_uptime_ms(self, signal_state: TrackingSignalState) -> tuple[float, float]:
        code_phase_ms = signal_state.code_phase_ms
        code_rate_ms_per_sec = signal_state.code_rate_ms_per_sec
        start_uptime_ms = (
            self.start_code_phase_ms - code_phase_ms
        ) / code_rate_ms_per_sec * 1e3 + signal_state.uptime_epoch_ms
        stop_uptime_ms = (
            self.stop_code_phase_ms - code_phase_ms
        ) / code_rate_ms_per_sec * 1e3 + signal_state.uptime_epoch_ms
        return start_uptime_ms, stop_uptime_ms


class TrackingChannel:
    """
    Tracking channel architecture:
    1. AlignedCorrelator accumulates and emits code-aligned 1 ms correlations.
    2. TrackingChannel consumes COMPLETE results only.
    3. Loop discriminators + filters update carrier/code to reflect state at start of correlation epochs.
    """

    def __init__(
        self,
        loop_params: TrackingLoopParameters,
        signal_params: TrackingSignalParameters,
        initial_signal_state: TrackingSignalState,
        output_capacity: int = 60000,
    ) -> None:
        self.loop_params = loop_params
        self.signal_params = signal_params

        self.signal_state = initial_signal_state
        self.loop_state = TrackingLoopState(mode=TrackingLoopMode.FLL)
        self.outputs = SignalTrackingOutputs(capacity=output_capacity)

        corr_config = DelayDopplerCorrelatorConfig(
            num_delays=3,
            delay_offset_chips=self.loop_params.EPL_chip_spacing,
            delay_step_chips=-self.loop_params.EPL_chip_spacing,
            num_dopplers=1,
            doppler_offset_hz=0.0,
            doppler_step_hz=0.0,
        )
        self.correlator = AlignedCorrelator(
            corr_config,
        )
        self.correlator_status = CorrelatorStatus.CLEARED
        start_code_phase_ms, _ = self.signal_state.propagate_phase(self.signal_state.uptime_epoch_ms)
        self.corr_interval = CorrelationInterval(
            start_code_phase_ms=int(start_code_phase_ms),
            duration_ms=self.loop_params.corr_period_ms,
        )


        # Hidden option/flag variables
        self._ignore_loop_updates = False


    def run_loop_filter(
        self,
    ) -> None:
        
        # Update signal state to reflect parameters at current correlation epoch
        uptime_epoch_ms = self.corr_interval.start_code_phase_ms
        dt_sec = (uptime_epoch_ms - self.signal_state.uptime_epoch_ms) * 1e-3

        # Assume single doppler for now, so take first column of correlator grid
        early = self.correlator.corr_grid[0, 0]
        prompt = self.correlator.corr_grid[1, 0]
        late = self.correlator.corr_grid[2, 0]

        if np.real(prompt) == 0.0:
            delta_theta = 0.25
        else:
            delta_theta = np.arctan(np.imag(prompt) / np.real(prompt)) / (2.0 * np.pi)

        last_prompt = self.loop_state.get_last_prompt_corr()
        if last_prompt is None or last_prompt == 0.0:
            delta_omega = 0.0
        else:
            delta_omega = np.angle(prompt / last_prompt) / (2.0 * np.pi * dt_sec)

        epl = self.loop_params.EPL_chip_spacing
        denom = np.abs(early) + np.abs(late) + 2.0 * np.abs(prompt)
        if denom < 1e-12:
            delta_eta = 0.0
        else:
            delta_eta = 0.5 * epl * (np.abs(early) - np.abs(late)) / denom

        self.loop_state.update_history(prompt)
        circ_length = self.loop_state.compute_prompt_corr_history_circ_length(costas=True)

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

        carrier_phase_cycles = (
            self.signal_state.carrier_phase_cycles
            + self.signal_state.carrier_rate_cyc_per_sec * dt_sec
            + filt_carr_phase_error_cycles
        )
        doppler_freq_hz = self.signal_state.carrier_rate_cyc_per_sec + filt_doppler_freq_error_hz

        code_phase_chips = (
            self.signal_state.code_phase_ms
            * 1e-3
            * self.signal_params.nominal_code_rate_chips_per_sec
        )
        code_rate_chips_per_sec = self.signal_params.nominal_code_rate_chips_per_sec * (
            1.0 + doppler_freq_hz / self.signal_params.carrier_freq_hz
        )
        code_phase_chips += code_rate_chips_per_sec * dt_sec + filt_code_phase_error_chips

        self.signal_state.uptime_epoch_ms = uptime_epoch_ms
        self.signal_state.carrier_phase_cycles = carrier_phase_cycles
        self.signal_state.carrier_rate_cyc_per_sec = doppler_freq_hz
        self.signal_state.code_phase_ms = (
            1e3 * code_phase_chips / self.signal_params.nominal_code_rate_chips_per_sec
        )

        idx = self.outputs.output_index
        if idx < len(self.outputs.uptime_epoch_ms):
            self.outputs.uptime_epoch_ms[idx] = uptime_epoch_ms
            self.outputs.carr_phase_errors_cycles[idx] = delta_theta
            self.outputs.code_phase_errors_chips[idx] = delta_eta
            self.outputs.early_corr[idx] = early
            self.outputs.prompt_corr[idx] = prompt
            self.outputs.late_corr[idx] = late
            self.outputs.carr_phase_cycles[idx] = carrier_phase_cycles
            self.outputs.doppler_freq_hz[idx] = doppler_freq_hz
            self.outputs.code_phase_ms[idx] = self.signal_state.code_phase_ms
            self.outputs.delta_omega[idx] = delta_omega
            self.outputs.prompt_corr_circ_length[idx] = circ_length
            self.outputs.output_index += 1


    def process_sample_buffer(
            self, 
            buffer: sample_streaming.SampleBuffer
        ) -> None:

        while True:
            
            # Determine code period integration bounds for current signal state
            corr_interval_start_uptime_ms, corr_interval_stop_uptime_ms = self.corr_interval.compute_start_and_stop_uptime_ms(
                self.signal_state
            )

            # If previous correlation interval status was partial, check whether this will complete the interval
            # If it will not, then we can ignore the last correlation (reset correlator) and start new accum.
            if self.correlator_status == CorrelatorStatus.PARTIAL and buffer.start_uptime_ms > corr_interval_stop_uptime_ms:
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
                self.correlator_status = CorrelatorStatus.COMPLETE
                self.run_loop_filter()

                self.corr_interval.increment()
                self.correlator.reset()
                self.correlator_status = CorrelatorStatus.CLEARED
            else:
                self.correlator_status = CorrelatorStatus.PARTIAL
                break