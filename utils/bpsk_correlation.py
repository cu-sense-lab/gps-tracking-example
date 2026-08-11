import numpy as np
import numba as nb
from numpy.typing import NDArray

@nb.jit(nopython=True, parallel=False)
def numba_correlate__bpsk__complex64(
        samples: nb.complex64[:],  # type: ignore
        code_seq: nb.int8[:],  # type: ignore
        code_length: nb.int32,  # type: ignore
        chip_start: nb.float32,  # type: ignore
        chip_delta: nb.float32,  # type: ignore
        num_bins: nb.int32,  # type: ignore
        chip_bin_offset: nb.float32,  # type: ignore
        chip_bin_spacing: nb.float32,  # type: ignore
        conj_carr_sample: nb.complex64,  # type: ignore
        conj_carr_rotation: nb.complex64,  # type: ignore
        corr_values: nb.complex64[:]  # type: ignore
    ) -> None:
    
    num_samples = len(samples)
    center_chip = chip_start
    for i in range(num_samples):
        carrierless = samples[i] * conj_carr_sample
        for j in range(num_bins):
            symbol = code_seq[int(center_chip + chip_bin_offset + j * chip_bin_spacing) % code_length]
            if symbol == 1:
                corr_values[j] += carrierless
            elif symbol == -1:
                corr_values[j] -= carrierless
            elif symbol != 0:
                corr_values[j] += carrierless * symbol
        conj_carr_sample *= conj_carr_rotation
        center_chip += chip_delta


def correlate__delay(
    samples: np.ndarray,
    samp_rate: float,
    initial_carr_phase_cycles: float,
    doppler_freq_hz: float,
    code_seq: np.ndarray,
    code_length_chips: int,
    code_rate_chips_per_sec: float,
    initial_code_phase_chips: float,
    num_chip_bins: int,
    chip_bin_offset: float,
    chip_bin_spacing: float,
    output: NDArray[np.complex64]
) -> None:

    chip_start = initial_code_phase_chips % code_length_chips
    chip_delta = code_rate_chips_per_sec / samp_rate  # chips per sample

    conj_carr_sample = np.exp(-2j * np.pi * initial_carr_phase_cycles)
    conj_carr_rotation = np.exp(-2j * np.pi * doppler_freq_hz / samp_rate)

    numba_correlate__bpsk__complex64(
        samples,
        code_seq,
        code_length_chips,
        chip_start,
        chip_delta,
        num_chip_bins,
        chip_bin_offset,
        chip_bin_spacing,
        conj_carr_sample,
        conj_carr_rotation,
        output
    )


@nb.jit(nopython=True, parallel=False)
def numba_correlate__interleaved_bpsk__complex64(
        samples: nb.complex64[:],  # type: ignore
        code_seq_0: nb.int8[:],  # type: ignore
        code_seq_1: nb.int8[:],  # type: ignore
        code_length_0: nb.int32,  # type: ignore
        code_length_1: nb.int32,  # type: ignore
        chip_start: nb.float32,  # type: ignore
        chip_delta: nb.float32,  # type: ignore
        num_bins: nb.int32,  # type: ignore
        chip_bin_offset: nb.float32,  # type: ignore
        chip_bin_spacing: nb.float32,  # type: ignore
        conj_carr_sample: nb.complex64,  # type: ignore
        conj_carr_rotation: nb.complex64,  # type: ignore
        corr_values: nb.complex64[:, :]  # type: ignore
    ) -> None:
    
    num_samples = len(samples)
    center_chip = chip_start
    for i in range(num_samples):
        carrierless = samples[i] * conj_carr_sample
        for j in range(num_bins):
            chip_index = int(center_chip + chip_bin_offset + j * chip_bin_spacing)
            # chip_index advances at the combined (interleaved) chip rate. Each component
            # occupies every other chip, so its own index into its code is chip_index // 2.
            component_chip_index = chip_index // 2
            if chip_index % 2 == 0:
                symbol = code_seq_0[component_chip_index % code_length_0]
                if symbol == 1:
                    corr_values[j, 0] += carrierless
                elif symbol == -1:
                    corr_values[j, 0] -= carrierless
                elif symbol != 0:
                    corr_values[j, 0] += carrierless * symbol
            else:
                symbol = code_seq_1[component_chip_index % code_length_1]
                if symbol == 1:
                    corr_values[j, 1] += carrierless
                elif symbol == -1:
                    corr_values[j, 1] -= carrierless
                elif symbol != 0:
                    corr_values[j, 1] += carrierless * symbol
        conj_carr_sample *= conj_carr_rotation
        center_chip += chip_delta


def correlate_delay_interleaved(
    samples: np.ndarray,
    samp_rate: float,
    initial_carr_phase_cycles: float,
    doppler_freq_hz: float,
    code_seq_0: np.ndarray,
    code_seq_1: np.ndarray,
    code_length_0: int,
    code_length_1: int,
    code_rate_chips_per_sec: float,
    initial_code_phase_chips: float,
    num_chip_bins: int,
    chip_bin_offset: float,
    chip_bin_spacing: float,
    output: NDArray[np.complex64]
) -> None:

    # The interleaved stream repeats every 2 * lcm(len0, len1) combined chips, which is
    # 2 * max(...) for a divisor pair.  NOTE: code lengths must be divisor pair.
    # Reducing by anything else (e.g. max alone) shifts both components' chip_index // 2
    # and breaks alignment once the code phase runs past one code period.
    interleaved_period_chips = 2 * max(code_length_0, code_length_1)
    chip_start = initial_code_phase_chips % interleaved_period_chips
    chip_delta = code_rate_chips_per_sec / samp_rate  # chips per sample

    conj_carr_sample = np.exp(-2j * np.pi * initial_carr_phase_cycles)
    conj_carr_rotation = np.exp(-2j * np.pi * doppler_freq_hz / samp_rate)

    numba_correlate__interleaved_bpsk__complex64(
        samples,
        code_seq_0,
        code_seq_1,
        code_length_0,
        code_length_1,
        chip_start,
        chip_delta,
        num_chip_bins,
        chip_bin_offset,
        chip_bin_spacing,
        conj_carr_sample,
        conj_carr_rotation,
        output
    )