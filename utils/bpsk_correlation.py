import numpy as np
import numba as nb

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
) -> np.ndarray[np.complex64]:
        
    chip_start = initial_code_phase_chips % code_length_chips
    chip_delta = code_rate_chips_per_sec / samp_rate  # chips per sample

    conj_carr_sample = np.exp(-2j * np.pi * initial_carr_phase_cycles)
    conj_carr_rotation = np.exp(-2j * np.pi * doppler_freq_hz / samp_rate)

    output = np.zeros(num_chip_bins, dtype=np.complex64)

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

    return output