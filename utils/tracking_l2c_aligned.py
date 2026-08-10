from dataclasses import dataclass

import numpy as np

from utils import sample_streaming

from .bpsk_correlation import correlate_delay_interleaved
from .tracking_bpsk_aligned import (
    CorrelationInterval,
    CorrelatorStatus,
    DelayDopplerCorrelatorConfig,
    TrackingLoopMode,
    TrackingLoopParameters,
    TrackingLoopState,
    TrackingSignalState,
)


@dataclass
class L2CTrackingSignalParameters:
    code_seq_0: np.ndarray
    code_seq_1: np.ndarray
    nominal_code_rate_chips_per_sec: float
    carrier_freq_hz: float

    @property
    def code_0_length_chips(self) -> int:
        return len(self.code_seq_0)

    @property
    def code_1_length_chips(self) -> int:
        return len(self.code_seq_1)


class L2CAlignedCorrelator:
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
        self.corr_grid = np.zeros((config.num_delays, config.num_dopplers, 2), dtype=np.complex64)
        # Every delay/doppler bin accumulates the same number of samples per call,
        # so a single running count is sufficient (was a same-valued grid before).
        self.corr_count = 0

    def reset(self) -> None:
        self.corr_grid.fill(0.0)
        self.corr_count = 0

    def accumulate(
        self,
        buffer: sample_streaming.SampleBuffer,
        accum_start_uptime_ms: float,
        accum_stop_uptime_ms: float,
        signal_params: L2CTrackingSignalParameters,
        signal_state: TrackingSignalState,
    ) -> None:

        if not (accum_start_uptime_ms < accum_stop_uptime_ms):
            raise ValueError("accumulation stop must be after accumulation start")

        # Determine start and end samples, trimmed to the buffer.
        # Bounds have to be both floor or ceiling; ceiling gives the exact half-open
        # window [start, stop), so consecutive intervals partition the stream.
        accum_start_sample_index = max(0, int(np.ceil((accum_start_uptime_ms - buffer.start_uptime_ms) * buffer.samp_rate / 1000)))
        accum_stop_sample_index = min(len(buffer.samples), int(np.ceil((accum_stop_uptime_ms - buffer.start_uptime_ms) * buffer.samp_rate / 1000)))
        if accum_start_sample_index >= accum_stop_sample_index:
            # The window does not overlap this buffer: it can begin inside the final
            # sample period, or be shorter than one sample period. The channel will mark
            # the interval PARTIAL and resume from the interval start on the next buffer.
            return

        actual_accum_start_uptime_ms = buffer.start_uptime_ms + accum_start_sample_index / buffer.samp_rate * 1000
        samples = buffer.samples[accum_start_sample_index:accum_stop_sample_index]
        num_accum_samples = len(samples)

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
            correlate_delay_interleaved(
                samples,
                buffer.samp_rate,
                corr_carrier_phase_cycles,
                corr_doppler_hz,
                signal_params.code_seq_0,
                signal_params.code_seq_1,
                signal_params.code_0_length_chips,
                signal_params.code_1_length_chips,
                code_rate_chips_per_sec,
                code_phase_chips,
                self.config.num_delays,
                self.config.delay_offset_chips,
                self.config.delay_step_chips,
                self.corr_grid[:, i_dopp, :],
            )
        self.corr_count += num_accum_samples


@dataclass
class L2CSignalTrackingOutputs:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.uptime_epoch_ms = np.zeros(capacity, dtype=float)
        self.carr_phase_errors_cycles = np.zeros(capacity, dtype=float)
        self.code_phase_errors_chips = np.zeros(capacity, dtype=float)
        # Column 0 is CM, column 1 is CL.
        self.early_corr = np.zeros((capacity, 2), dtype=complex)
        self.prompt_corr = np.zeros((capacity, 2), dtype=complex)
        self.late_corr = np.zeros((capacity, 2), dtype=complex)
        self.carr_phase_cycles = np.zeros(capacity, dtype=float)
        self.doppler_freq_hz = np.zeros(capacity, dtype=float)
        self.code_phase_ms = np.zeros(capacity, dtype=float)
        self.delta_omega = np.zeros(capacity, dtype=float)
        self.prompt_corr_circ_length = np.zeros(capacity, dtype=float)
        self.output_index = 0


class TrackingChannel:
    """
    Tracking channel architecture:
    1. L2CAlignedCorrelator accumulates and emits code-aligned 1 ms correlations.
    2. TrackingChannel consumes COMPLETE results only.
    3. Loop discriminators + filters update carrier/code to reflect state at start of correlation epochs.

    Only the CM component drives the discriminators; CL is accumulated and stored
    but does not close the loop.
    """

    def __init__(
        self,
        loop_params: TrackingLoopParameters,
        signal_params: L2CTrackingSignalParameters,
        initial_signal_state: TrackingSignalState,
        output_capacity: int = 60000,
    ) -> None:
        self.loop_params = loop_params
        self.signal_params = signal_params

        self.signal_state = initial_signal_state
        self.loop_state = TrackingLoopState(mode=TrackingLoopMode.FLL)
        self.outputs = L2CSignalTrackingOutputs(capacity=output_capacity)

        corr_config = DelayDopplerCorrelatorConfig(
            num_delays=3,
            delay_offset_chips=self.loop_params.EPL_chip_spacing,
            delay_step_chips=-self.loop_params.EPL_chip_spacing,
            num_dopplers=1,
            doppler_offset_hz=0.0,
            doppler_step_hz=0.0,
        )
        self.correlator = L2CAlignedCorrelator(
            corr_config,
        )
        self.correlator_status = CorrelatorStatus.CLEARED
        start_code_phase_ms, _ = self.signal_state.propagate_phase(self.signal_state.uptime_epoch_ms)
        self.corr_interval = CorrelationInterval(
            start_code_phase_ms=int(np.ceil(start_code_phase_ms)),
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

        # Grid is (delay, doppler, component); assume single doppler for now, so take first column.
        # Keep the full component vectors for output, and drive the loops with CM (component 0).
        early_CM_CL, prompt_CM_CL, late_CM_CL = self.correlator.corr_grid[:, 0]
        early = early_CM_CL[0]
        prompt = prompt_CM_CL[0]
        late = late_CM_CL[0]

        # Compute loop discriminators
        # Phase discriminator for PLL
        delta_theta = np.mod(np.angle(prompt) / (2.0 * np.pi) + 0.25, 0.5) - 0.25

        # Frequency discriminator for FLL
        last_prompt = self.loop_state.get_last_prompt_corr()
        if last_prompt is None or last_prompt == 0.0:
            delta_omega = 0.0
        else:
            # Costas-wrapped to +/- 1/4 cycle so nav-bit flips are not read as
            # frequency error, then divided by the epoch spacing to give Hz.
            delta_omega = (np.mod(np.angle(prompt / last_prompt) / (2.0 * np.pi) + 0.25, 0.5) - 0.25) / dt_sec

        # Code discriminator for DLL
        EPL_chip_spacing = self.loop_params.EPL_chip_spacing
        denom = np.abs(early) + np.abs(late) + 2.0 * np.abs(prompt)
        if denom < 1e-12:
            delta_eta = 0.0
        else:
            delta_eta = (2 - EPL_chip_spacing) * (np.abs(early) - np.abs(late)) / denom

        # Apply loop filters to discriminators
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
            + filt_code_phase_error_chips / self.signal_params.nominal_code_rate_chips_per_sec * 1e3
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
            self.outputs.early_corr[idx] = early_CM_CL
            self.outputs.prompt_corr[idx] = prompt_CM_CL
            self.outputs.late_corr[idx] = late_CM_CL
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
