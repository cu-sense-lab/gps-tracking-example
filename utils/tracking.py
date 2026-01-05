import numpy as np
from dataclasses import dataclass
from .bpsk_correlation import correlate__delay
from enum import Enum


@dataclass
class TrackingSignalState:
    uptime_epoch_seconds: float  # time at which state is valid
    code_phase_seconds: float
    carrier_phase_cycles: float
    doppler_freq_hz: float


class TrackingLoopMode(Enum):
    FLL = "FLL"
    PLL = "PLL"


@dataclass
class TrackingLoopState:
    mode: TrackingLoopMode
    history_size: int = 10

    def __post_init__(self):
        # self.delta_theta_history = np.zeros(self.history_size, dtype=float)
        self.prompt_corr_history = np.zeros(self.history_size, dtype=complex)
        self.history_index = 0
        self.history_filled = False

    def update_history(
        self,
        # delta_theta: float,
        prompt_corr: complex,
    ) -> None:
        # self.delta_theta_history[self.history_index] = delta_theta
        self.prompt_corr_history[self.history_index] = prompt_corr
        self.history_index += 1
        if self.history_index >= self.history_size:
            self.history_index = 0
            self.history_filled = True

    # @property
    # def last_delta_theta(self) -> float:
    #     if self.history_index == 0 and self.history_filled:
    #         return self.delta_theta_history[-1]
    #     else:
    #         return self.delta_theta_history[self.history_index - 1]

    @property
    def last_prompt_corr(self) -> complex:
        if self.history_index == 0:
            if self.history_filled:
                return self.prompt_corr_history[-1]
            # else:
            #     return 1.0 + 0.0j  # avoid zero division
        else:
            return self.prompt_corr_history[self.history_index - 1]

    def compute_prompt_corr_history_circ_length(self, costas: bool = False) -> float:
        if self.history_filled:
            iq = self.prompt_corr_history
        else:
            iq = self.prompt_corr_history[: self.history_index]
        angles = np.angle(iq)
        if costas:
            angles = 2.0 * angles
        circ_length = np.abs(np.mean(np.exp(1j * angles)))
        return circ_length


@dataclass
class TrackingSignalParameters:
    code_seq: np.ndarray[np.int8]
    nominal_code_rate_chips_per_sec: float
    carrier_freq_hz: float

    @property
    def code_length_chips(self) -> int:
        return len(self.code_seq)


@dataclass
class SignalTrackingOutputs:

    def __init__(self, output_capacity: int):
        self.uptime_seconds = np.zeros(output_capacity, dtype=float)
        self.carr_phase_errors_cycles = np.zeros(output_capacity, dtype=float)
        self.code_phase_errors_chips = np.zeros(output_capacity, dtype=float)
        self.early_corr = np.zeros(output_capacity, dtype=complex)
        self.prompt_corr = np.zeros(output_capacity, dtype=complex)
        self.late_corr = np.zeros(output_capacity, dtype=complex)
        self.carr_phase_cycles = np.zeros(output_capacity, dtype=float)
        self.doppler_freq_hz = np.zeros(output_capacity, dtype=float)
        self.code_phase_seconds = np.zeros(output_capacity, dtype=float)

        self.delta_omega = np.zeros(output_capacity, dtype=float)
        self.prompt_corr_circ_length = np.zeros(output_capacity, dtype=float)

        self.output_index = 0


@dataclass
class TrackingLoopParameters:
    DLL_bandwidth_hz: float
    PLL_bandwidth_hz: float
    FLL_bandwidth_hz: float
    update_period_ms: float
    block_duration_ms: int
    samp_rate: float
    EPL_chip_spacing: float = 0.5  # chips
    prompt_corr_circ_length_threshold = 0.9

    def __post_init__(self):
        # Compute block sample time array
        block_duration_seconds = self.block_duration_ms * 1e-3
        num_samples_per_block = int(self.samp_rate * block_duration_seconds)
        self.block_sample_time_arr = np.arange(num_samples_per_block) / self.samp_rate
        # Compute loop filter coefficients
        update_period_seconds = self.update_period_ms * 1e-3
        self.DLL_filter_coeff = 4 * update_period_seconds * self.DLL_bandwidth_hz
        xi = 1 / np.sqrt(2)
        omega_n = self.PLL_bandwidth_hz / 0.53
        self.PLL_filter_coeffs = (
            2 * xi * omega_n * update_period_seconds
            - 3 / 2 * omega_n**2 * update_period_seconds**2,
            omega_n**2 * update_period_seconds,
        )

        # if FLL bandwidth is the equivalent noise bandwidth of the filter, then
        self.FLL_filter_coeff = 4 * self.FLL_bandwidth_hz * update_period_seconds


def track_signal(
    uptime_seconds: float,
    sample_block: np.ndarray,
    signal_params: TrackingSignalParameters,
    signal_state: TrackingSignalState,  # *IO
    loop_params: TrackingLoopParameters,
    loop_state: TrackingLoopState,  # *IO
    signal_outputs: SignalTrackingOutputs,  # *IO
) -> None:
    # Unpack values
    code_seq = signal_params.code_seq
    nominal_code_rate_chips_per_sec = signal_params.nominal_code_rate_chips_per_sec
    code_length_chips = signal_params.code_length_chips
    carrier_freq_hz = signal_params.carrier_freq_hz

    carrier_phase_cycles = signal_state.carrier_phase_cycles
    doppler_freq_hz = signal_state.doppler_freq_hz
    code_phase_chips = signal_state.code_phase_seconds * nominal_code_rate_chips_per_sec
    adjusted_code_rate_chips_per_sec = nominal_code_rate_chips_per_sec * (
        1 + doppler_freq_hz / carrier_freq_hz
    )

    # Propagate carrier and code phases to start of block
    time_delta = uptime_seconds - signal_state.uptime_epoch_seconds
    carrier_phase_cycles += doppler_freq_hz * time_delta
    code_phase_chips += adjusted_code_rate_chips_per_sec * time_delta

    # Perform correlation
    # SLOW VERSION:
    # First perform carrier wipeoff
    # phi = 2.0 * np.pi * (carrier_phase_cycles + doppler_freq_hz * loop_params.block_sample_time_arr)
    # wipeoff_samples = sample_block * np.exp(-1j * phi)
    # # Next perform code correlation (early, prompt, late)
    # prompt_code_phase_chips = code_phase_chips + loop_params.block_sample_time_arr * adjusted_code_rate_chips_per_sec
    # corr_values = np.zeros(3, dtype=complex)
    # EPL_chip_spacing = loop_params.EPL_chip_spacing
    # for i_bin, chip_bin_offset in enumerate([EPL_chip_spacing, 0.0, -EPL_chip_spacing]):
    #     offset_code_phase_chips = prompt_code_phase_chips + chip_bin_offset
    #     chip_indices = np.floor(offset_code_phase_chips).astype(int) % code_length_chips
    #     code_samples = code_seq[chip_indices]
    #     corr_values[i_bin] = np.sum(wipeoff_samples * np.conj(code_samples))
    # early, prompt, late = corr_values

    # # FAST VERSION:
    EPL_chip_spacing = loop_params.EPL_chip_spacing
    early, prompt, late = correlate__delay(
        sample_block,
        loop_params.samp_rate,
        carrier_phase_cycles,
        doppler_freq_hz,
        code_seq,
        code_length_chips,
        adjusted_code_rate_chips_per_sec,
        code_phase_chips,
        3,
        EPL_chip_spacing,
        -EPL_chip_spacing,
    )

    # Estimate state errors from correlator outputs
    # delta_theta = np.angle(prompt) / (2.0 * np.pi)
    # # Costas (arctan) discriminator for carrier phase error
    # if delta_theta > 0.5:
    #     delta_theta -= 1.0
    # elif delta_theta < -0.5:
    #     delta_theta += 1.0
    if np.real(prompt) == 0.0:
        delta_theta = 0.25  # avoid zero division
    else:
        delta_theta = np.arctan(np.imag(prompt) / np.real(prompt)) / (2.0 * np.pi)

    # FLL discriminator for carrier freq error
    last_prompt_corr = loop_state.last_prompt_corr
    if last_prompt_corr == 0.0:
        delta_omega = 0.0  # avoid zero division
    else:
        delta_omega = np.angle(prompt / loop_state.last_prompt_corr) / (2e-3 * np.pi * loop_params.block_duration_ms)
    # DLL discriminator for code phase error
    delta_eta = (
        0.5
        * EPL_chip_spacing
        * (np.abs(early) - np.abs(late))
        / (np.abs(early) + np.abs(late) + 2 * np.abs(prompt))
    )

    loop_state.update_history(prompt)
    circ_length = loop_state.compute_prompt_corr_history_circ_length(costas=True)

    # Apply loop filter to errors
    # DLL
    filt_code_phase_error_chips = loop_params.DLL_filter_coeff * delta_eta
    # PLL/FLL
    if loop_state.mode == TrackingLoopMode.FLL:
        # FLL to PLL transition condition
        if (
            loop_state.history_filled
            and circ_length > loop_params.prompt_corr_circ_length_threshold
        ):
            loop_state.mode = TrackingLoopMode.PLL
        # FLL update (simple proportional)
        # IQ2 / IQ1 -> IQ2 * conj(IQ1) / |IQ1|^2 = ((I2 * I1 + Q2 * Q1) - j * (I2 * Q1 + Q2 * I1)) / (I1^2 + Q1^2)
        filt_carr_phase_error_cycles = 0.0
        filt_doppler_freq_error_hz = loop_params.FLL_filter_coeff * delta_omega
    else:
        # PLL update (2nd order loop)
        filt_carr_phase_error_cycles = loop_params.PLL_filter_coeffs[0] * delta_theta
        filt_doppler_freq_error_hz = loop_params.PLL_filter_coeffs[1] * delta_theta

    # Update signal state
    carrier_phase_cycles += filt_carr_phase_error_cycles
    doppler_freq_hz += filt_doppler_freq_error_hz
    code_phase_chips += filt_code_phase_error_chips

    signal_state.uptime_epoch_seconds = uptime_seconds
    signal_state.carrier_phase_cycles = carrier_phase_cycles
    signal_state.doppler_freq_hz = doppler_freq_hz
    signal_state.code_phase_seconds = code_phase_chips / nominal_code_rate_chips_per_sec

    # Store outputs
    output_idx = signal_outputs.output_index
    signal_outputs.uptime_seconds[output_idx] = uptime_seconds
    signal_outputs.carr_phase_errors_cycles[output_idx] = delta_theta
    signal_outputs.code_phase_errors_chips[output_idx] = delta_eta
    signal_outputs.early_corr[output_idx] = early
    signal_outputs.prompt_corr[output_idx] = prompt
    signal_outputs.late_corr[output_idx] = late
    signal_outputs.carr_phase_cycles[output_idx] = carrier_phase_cycles
    signal_outputs.doppler_freq_hz[output_idx] = doppler_freq_hz
    signal_outputs.code_phase_seconds[output_idx] = signal_state.code_phase_seconds
    signal_outputs.prompt_corr_circ_length[output_idx] = circ_length
    signal_outputs.delta_omega[output_idx] = delta_omega
    signal_outputs.output_index += 1


class TrackingChannel:

    def __init__(
        self,
        loop_params: TrackingLoopParameters,
        signal_params: TrackingSignalParameters,
        initial_uptime_seconds: float = 0.0,
        initial_code_phase_seconds: float = 0.0,
        initial_carrier_phase_cycles: float = 0.0,
        initial_doppler_freq_hz: float = 0.0,
        output_capacity: int = 60000,  # 1 minute at 1 ms blocks
    ) -> None:
        self.loop_params = loop_params
        self.loop_state = TrackingLoopState(mode=TrackingLoopMode.FLL)
        self.signal_params = signal_params
        self.signal_state = TrackingSignalState(
            uptime_epoch_seconds=initial_uptime_seconds,
            code_phase_seconds=initial_code_phase_seconds,
            carrier_phase_cycles=initial_carrier_phase_cycles,
            doppler_freq_hz=initial_doppler_freq_hz,
        )
        self.outputs = SignalTrackingOutputs(output_capacity=output_capacity)

    def process_sample_block(
        self,
        uptime_seconds: float,
        sample_block: np.ndarray,
    ) -> None:
        track_signal(
            uptime_seconds,
            sample_block,
            self.signal_params,
            self.signal_state,
            self.loop_params,
            self.loop_state,
            self.outputs,
        )
