import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
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
    sequence: np.ndarray[np.int8]


@dataclass
class CorrelationResult:
    correlation_matrix: np.ndarray[np.float64]
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
    def doppler_bins_hz(self) -> np.ndarray[np.float64]:
        return (
            self.start_doppler_hz
            + np.arange(self.num_doppler_bins) * self.doppler_resolution_hz
        )

    @property
    def code_phase_bins_seconds(self) -> np.ndarray[np.float64]:
        return (
            self.start_code_phase_seconds
            + np.arange(self.num_code_phase_bins) * self.code_phase_resolution_seconds
        )


@dataclass
class AcquisitionResult:
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

    @property
    def acq_doppler_hz(self) -> float:
        return (
            self.correlation_result.start_doppler_hz
            + self.peak_doppler_bin * self.correlation_result.doppler_resolution_hz
        )

    @property
    def acq_code_phase_seconds(self) -> float:
        return (
            self.correlation_result.start_code_phase_seconds
            + self.peak_code_phase_bin
            * self.correlation_result.code_phase_resolution_seconds
        )


def run_acquisition(
    sample_block: np.ndarray[np.complex64],
    acq_config: AcquisitionConfiguration,
    code_parameters: Dict[str, AcqSignalCodeParameters],
    prob_false_alaram: float,
    print_progress: bool = False,
) -> Dict[str, AcquisitionResult]:
    """
    Perform BPSK acquisition on the given sample block for all signals defined in code_parameters.

    Returns a dictionary mapping signal IDs to their respective AcquisitionResult.

    Pre-computed replicas and FFTs are stored for each signal in the config `replica_cache_dict`.
    The values in the cache entries must be recomputed when acquisition parameters change or when the sampling rate changes.
    """
    acquisition_results: Dict[str, AcquisitionResult] = {}

    #
    # Reshape sample block into M blocks of N samples
    M = acq_config.num_blocks
    N = acq_config.block_size_samples
    samples = sample_block[: M * N].reshape((M, N))
    # Compute FFT of blocks
    samples_fft = np.fft.fft(samples, axis=1)

    for signal_id, code_params in code_parameters.items():

        if print_progress:
            print(f"Acquiring signal {signal_id}...", end="")

        # Check if there is a cached replica for this signal
        if signal_id in acq_config.replica_cache_dict:
            replica_entry = acq_config.replica_cache_dict[signal_id]
            replica_samples_fft = replica_entry.replica_fft
        else:
            replica_samples = np.zeros(
                acq_config.block_size_samples, dtype=np.complex64
            )
            chips_arr = (
                0.0 + acq_config.replica_time_arr * code_params.rate_chips_per_sec
            )
            chip_indices = chips_arr.astype(int) % code_params.length_chips
            replica_samples[: acq_config.replica_length_samples] = code_params.sequence[
                chip_indices
            ].astype(float)
            replica_samples_fft = np.fft.fft(replica_samples)
            replica_entry = SignalReplicaCacheEntry(
                replica_samples, replica_samples_fft
            )
            acq_config.replica_cache_dict[signal_id] = replica_entry

        doppler_search_bins = acq_config.doppler_search_bins
        correlation = np.zeros(
            (len(doppler_search_bins), acq_config.block_size_samples)
        )

        for i, roll in enumerate(doppler_search_bins):
            # coherent integration over N samples; z_noise ~ CN(0, N*noise_var)
            # shifted_samples_fft = np.roll(samples_fft, -roll, axis=1)
            # corr = np.fft.ifft(
            #     np.conj(shifted_samples_fft) * replica_samples_fft[None, :]
            # )

            shifted_replica_fft = np.roll(replica_samples_fft, roll)
            corr = np.fft.ifft(
                np.conj(samples_fft) * shifted_replica_fft[None, :]
            )
            # non-coherent square-law summation over M blocks, normalized by N
            # y_noise / noise_var ~ ChiSquared(2M)
            correlation[i] = np.sum(1 / N * np.abs(corr) ** 2, axis=0)

        # Find acquisition peak
        peak_doppler_bin, peak_sample_bin = np.unravel_index(
            correlation.argmax(), correlation.shape
        )
        peak_val = correlation[peak_doppler_bin, peak_sample_bin]

        # Estimate noise distribution
        # y_noise = X * noise_var
        # E[y_noise] = 2 * M * noise_var
        # Var[y_noise] = 4 * M * noise_var**2
        # Don't worry about peak power, its find to overestimate noise a bit
        y_noise_mean = np.mean(correlation)
        noise_var = y_noise_mean / (2 * acq_config.num_blocks)
        # Another way to estimate noise stddev;  can be way off if strong signal present
        # y_noise_var = np.var(correlation)
        # noise = np.sqrt(y_noise_var / (4 * acq_config.num_blocks))

        normalized_peak_value = peak_val / noise_var
        chi2 = scipy.stats.chi2(df=2 * acq_config.num_blocks)
        detection_threshold = chi2.ppf(1 - prob_false_alaram)
        signal_detected = normalized_peak_value > detection_threshold

        corr_result = CorrelationResult(
            correlation,
            acq_config.doppler_search_bins[0] * acq_config.fft_resolution,
            acq_config.fft_resolution,
            0.0,
            1 / acq_config.sample_rate,
        )

        acq_result = AcquisitionResult(
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
        )

        acquisition_results[signal_id] = acq_result

        if print_progress:
            print(
                f"{normalized_peak_value:6.3f}{'*' if signal_detected else ''}",
                end="\n",
            )

    return acquisition_results
