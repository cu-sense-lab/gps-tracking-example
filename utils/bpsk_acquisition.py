import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from numpy.typing import NDArray
import scipy.stats


@dataclass
class SignalReplicaCacheEntry:
    replica_samples: np.ndarray
    replica_fft: np.ndarray


@dataclass
class AcquisitionConfiguration:
    replica_duration_ms: int
    num_blocks: int
    sample_rate: int
    min_search_doppler_hz: float
    max_search_doppler_hz: float

    def __post_init__(self):
        self.replica_length_samples = int(
            self.sample_rate * self.replica_duration_ms / 1000
        )
        self.replica_time_arr = (
            np.arange(self.replica_length_samples) / self.sample_rate
        )
        self.block_size_samples = self.replica_length_samples
        self.block_duration_seconds = self.block_size_samples / self.sample_rate
        self.fft_resolution = 1 / self.block_duration_seconds
        assert self.num_blocks > 0
        self.acq_total_duration_ms = self.num_blocks * self.replica_duration_ms
        self.total_num_samples = int(
            self.acq_total_duration_ms / 1000 * self.sample_rate
        )
        self.min_doppler_fft_bin = int(self.min_search_doppler_hz / self.fft_resolution)
        self.max_doppler_fft_bin = int(self.max_search_doppler_hz / self.fft_resolution)
        self.doppler_search_bins = np.arange(
            self.min_doppler_fft_bin, self.max_doppler_fft_bin
        )
        self.num_doppler_bins = len(self.doppler_search_bins)

        self.replica_cache_dict: Dict[str, SignalReplicaCacheEntry] = {}


@dataclass
class AcqSignalCodeParameters:
    rate_chips_per_sec: float
    length_chips: int
    sequence: NDArray[np.int8]
    # True for signals where `sequence` is one component of a chip-interleaved
    # pair (e.g. L2C's CM/CL). The replica is zero-filled on the other
    # component's chips rather than holding this component's chip value
    # across both, so it doesn't spuriously correlate against the other code.
    is_interleaved: bool = False


@dataclass
class CorrelationResult:
    correlation_matrix: NDArray[np.float64]
    start_doppler_hz: float
    doppler_resolution_hz: float
    start_code_phase_seconds: float
    code_phase_resolution_seconds: float

    @property
    def num_doppler_bins(self) -> int:
        return self.correlation_matrix.shape[0]

    @property
    def num_code_phase_bins(self) -> int:
        return self.correlation_matrix.shape[1]

    @property
    def doppler_bins_hz(self) -> NDArray[np.float64]:
        return np.asarray(
            self.start_doppler_hz
            + np.arange(self.num_doppler_bins) * self.doppler_resolution_hz,
            dtype=np.float64,
        )

    @property
    def code_phase_bins_seconds(self) -> NDArray[np.float64]:
        return np.asarray(
            self.start_code_phase_seconds
            + np.arange(self.num_code_phase_bins) * self.code_phase_resolution_seconds,
            dtype=np.float64,
        )


@dataclass
class AcquisitionResult:
    uptime_epoch_ms: float
    signal_id: str
    peak_doppler_bin: int
    peak_code_phase_bin: int
    normalized_peak_value: float
    prob_false_alarm: float
    detection_threshold: float
    noise_var: float
    signal_detected: bool
    correlation_result: CorrelationResult
    config: AcquisitionConfiguration
    # Sub-bin refinement, in Hz, from the half-bin Doppler search.  Zero when the
    # peak came from the plain integer-bin pass (and for all signals when the
    # search is disabled).
    doppler_offset_hz: float = 0.0

    @property
    def acq_doppler_hz(self) -> float:
        return (
            self.config.doppler_search_bins[self.peak_doppler_bin] * self.config.fft_resolution
            + self.doppler_offset_hz
        )

    @property
    def acq_code_phase_seconds(self) -> float:
        return (
            self.peak_code_phase_bin / self.config.sample_rate
        )


def run_acquisition(
    sample_block: NDArray[np.complex64],
    sample_block_uptime_epoch_ms: float,
    acq_config: AcquisitionConfiguration,
    code_parameters: Dict[str, AcqSignalCodeParameters],
    prob_false_alaram: float,
    print_progress: bool = False,
    noise_var_method: str = "abscorrmean",
    save_corr_peak_window_chips: Optional[float] = None,
    half_bin_doppler_search: bool = False,
) -> Dict[str, AcquisitionResult]:
    """
    Perform BPSK acquisition on the given sample block for all signals defined in code_parameters.

    Returns a dictionary mapping signal IDs to their respective AcquisitionResult.

    Pre-computed replicas and FFTs are stored for each signal in the config `replica_cache_dict`.
    The values in the cache entries must be recomputed when acquisition parameters change or when the sampling rate changes.

    Options for noise variance estimation:
        "abscorrmean": compute noise variance from the mean of abs(corr)**2 matrix
        "abscorrvar": compute noise variance from the square root of the variance of abs(corr)**2 matrix

    `half_bin_doppler_search` repeats the search with the samples shifted down by
    half a Doppler bin and keeps whichever pass peaks higher, halving the
    worst-case Doppler error at twice the cost.  It matters when the coherent
    integration length -- and therefore the bin width `1 / T_coherent` -- is
    forced short.  GPS L5 is the case in point: its Neuman-Hofman overlay caps
    coherent integration at 1 ms, giving 1000 Hz bins and up to +/-500 Hz of
    seeding error, while a 1 ms FLL discriminator is only unambiguous to
    +/-250 Hz.  Without this the tracking loop can be handed an error it cannot
    resolve.
    """
    acquisition_results: Dict[str, AcquisitionResult] = {}

    #
    # Reshape sample block into M blocks of N samples
    M = acq_config.num_blocks
    N = acq_config.block_size_samples
    samples = sample_block[: M * N].reshape((M, N))
    # Compute conjugated FFT of blocks, once per sub-bin Doppler hypothesis.
    # Blocks are combined non-coherently below, so applying the same phase ramp
    # from the start of each block (rather than from the start of the capture) is
    # sufficient -- only the frequency shift within a block matters.
    sub_bin_offsets_hz = [0.0]
    conj_sample_ffts = [np.conj(np.fft.fft(samples, axis=1))]
    if half_bin_doppler_search:
        half_bin_hz = 0.5 * acq_config.fft_resolution
        block_time_arr = np.arange(N) / acq_config.sample_rate
        shifted_samples = samples * np.exp(-2j * np.pi * half_bin_hz * block_time_arr)[None, :]
        conj_sample_ffts.append(np.conj(np.fft.fft(shifted_samples, axis=1)))
        # Mixing the samples down by half a bin makes the signal appear that much
        # lower, so the recovered Doppler is the bin centre plus the offset.
        sub_bin_offsets_hz.append(half_bin_hz)

    for signal_id, code_params in code_parameters.items():

        if print_progress:
            print(f"Acquiring signal {signal_id}...", end="")

        # Check if there is a cached replica for this signal
        if signal_id in acq_config.replica_cache_dict:
            replica_entry = acq_config.replica_cache_dict[signal_id]
            replica_samples_fft = replica_entry.replica_fft
        else:
            replica_samples = np.zeros(
                N, dtype=np.complex64
            )
            if code_params.is_interleaved:
                # Physical chip clock runs at 2x this component's own rate, alternating
                # between this component's chips (even chips) and the other's
                # (odd ones). Zero-fill those so the replica only correlates
                # against its own component, instead of smearing across both.
                physical_chip_idx = (
                    acq_config.replica_time_arr * 2.0 * code_params.rate_chips_per_sec
                ).astype(int)
                own_signal_chip = physical_chip_idx % 2 == 0
                own_chip_indices = (physical_chip_idx // 2) % code_params.length_chips
                replica_values = np.where(
                    own_signal_chip, code_params.sequence[own_chip_indices], 0
                )
            else:
                chips_arr = (
                    0.0 + acq_config.replica_time_arr * code_params.rate_chips_per_sec
                )
                chip_indices = chips_arr.astype(int) % code_params.length_chips
                replica_values = code_params.sequence[chip_indices]
            replica_samples[: acq_config.replica_length_samples] = replica_values.astype(float)
            replica_samples_fft = np.fft.fft(replica_samples)
            replica_entry = SignalReplicaCacheEntry(
                replica_samples, replica_samples_fft
            )
            acq_config.replica_cache_dict[signal_id] = replica_entry

        doppler_search_bins = acq_config.doppler_search_bins

        # Search each sub-bin Doppler hypothesis and keep whichever peaks highest.
        # With the half-bin search disabled there is exactly one hypothesis and
        # this reduces to the original single pass.
        correlation = None
        peak_doppler_bin = peak_sample_bin = 0
        peak_val = -np.inf
        doppler_offset_hz = 0.0

        for conj_samples_fft, sub_bin_offset_hz in zip(conj_sample_ffts, sub_bin_offsets_hz):
            pass_correlation = np.zeros((len(doppler_search_bins), N))

            for i, roll in enumerate(doppler_search_bins):
                # coherent integration over N samples; z_noise ~ CN(0, N*noise_var)
                shifted_replica_fft = np.roll(replica_samples_fft, roll)
                corr = np.fft.ifft(
                    conj_samples_fft * shifted_replica_fft[None, :]
                )
                # non-coherent square-law summation over M blocks, normalized by N
                # y_noise / noise_var ~ ChiSquared(2M)
                pass_correlation[i] = np.sum(1 / N * np.abs(corr) ** 2, axis=0)

            pass_doppler_bin_idx, pass_sample_bin_idx = np.unravel_index(
                pass_correlation.argmax(), pass_correlation.shape
            )
            pass_peak_val = pass_correlation[pass_doppler_bin_idx, pass_sample_bin_idx]
            if pass_peak_val > peak_val:
                correlation = pass_correlation
                peak_doppler_bin = int(pass_doppler_bin_idx)
                peak_sample_bin = int(pass_sample_bin_idx)
                peak_val = float(pass_peak_val)
                doppler_offset_hz = sub_bin_offset_hz

        assert correlation is not None  # at least one hypothesis is always searched

        # Estimate noise distribution
        # y_noise = X * noise_var
        # E[y_noise] = 2 * M * noise_var
        # Var[y_noise] = 4 * M * noise_var**2
        if noise_var_method == "abscorrmean":
            # Don't worry about peak power, its fine to overestimate noise a bit
            y_noise_mean = np.mean(correlation)
            noise_var = float(y_noise_mean / (2 * M))
        elif noise_var_method == "abscorrvar":
            # Another way to estimate noise stddev;  can be way off if strong signal present, but can be better estimate when lots of narrowband interference
            y_noise_var = np.var(correlation)
            noise_var = float(np.sqrt(y_noise_var / (4 * M)))
        else:
            raise ValueError(f"Unknown noise_var_method: {noise_var_method}")

        normalized_peak_value = peak_val / noise_var
        chi2 = scipy.stats.chi2(df=2 * M)
        detection_threshold = chi2.isf(prob_false_alaram)
        signal_detected = normalized_peak_value > detection_threshold

        # The reported Doppler axis belongs to the winning pass, so it carries the
        # same sub-bin offset as the peak.
        start_doppler_hz = (
            acq_config.doppler_search_bins[0] * acq_config.fft_resolution + doppler_offset_hz
        )
        if save_corr_peak_window_chips is not None:
            half_window_size_samples = int(
                save_corr_peak_window_chips / code_params.rate_chips_per_sec * acq_config.sample_rate
            )
            i0 = max(0, peak_sample_bin - half_window_size_samples)
            i1 = min(N, peak_sample_bin + half_window_size_samples)
            corr_result = CorrelationResult(
                correlation[:, i0:i1],
                start_doppler_hz,
                acq_config.fft_resolution,
                i0 / acq_config.sample_rate,
                1 / acq_config.sample_rate,
            )
        else:
            corr_result = CorrelationResult(
                correlation,
                start_doppler_hz,
                acq_config.fft_resolution,
                0.0,
                1 / acq_config.sample_rate,
            )

        acq_result = AcquisitionResult(
            sample_block_uptime_epoch_ms,
            signal_id,
            peak_doppler_bin,
            peak_sample_bin,
            normalized_peak_value,
            prob_false_alaram,
            detection_threshold,
            noise_var,
            signal_detected,
            corr_result,
            acq_config,
            doppler_offset_hz=doppler_offset_hz,
        )

        acquisition_results[signal_id] = acq_result

        if print_progress:
            print(
                f"{normalized_peak_value:6.3f}{'*' if signal_detected else ''}",
                end="\n",
            )

    return acquisition_results
