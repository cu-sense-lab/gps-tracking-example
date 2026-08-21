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
    """
    Acquisition dwell layout.

    Two lengths, and they are not the same thing:

    `coherent_duration_replica_ms` is the replica -- and therefore the FFT -- length.  The
    correlation is circular, so this must be a whole number of code periods or the
    wraparound matches the data against the wrong chips.

    `coherent_duration_sample_ms` is how much data goes into one coherent integration.
    It defaults to the replica length, which is what every caller did before it
    existed.  Setting it shorter zero-pads each block up to the replica length,
    and `num_blocks` of them are then summed by square law.

    Shortening it is how a signal whose data symbol is no longer than its code
    period gets acquired at all.  GPS L2 CM is the case in point: its 20 ms code
    period is exactly its CNAV symbol, so a 20 ms coherent integration at an
    arbitrary alignment straddles a symbol boundary.  There is a 50% chance of a
    flip, and when one happens the surviving amplitude is |2a - 1| for a boundary
    a fraction `a` of the way through -- a null when it lands mid-window.  Four
    5 ms blocks put at most one boundary inside any single block.

    The two lengths drive different things, which is the point of separating them:

        Doppler grid spacing   = 1 / replica_duration    (FFT bin width)
        Doppler response width = 1 / coherent_duration   (correlation mainlobe)

    With a short coherent length the grid is finer than the response, so the
    residual Doppler error is set by the grid rather than by the mainlobe.

    `fine_search_factors` is `(code_phase_factor, doppler_factor)`: how many
    sub-divisions of each search bin to test.  `(1, 4)` searches the code phase on
    the plain sample grid and the Doppler at quarter-bin spacing, cutting the
    worst-case Doppler error to a quarter.  `None` means `(1, 1)` -- one
    hypothesis per bin, the plain search.  Cost scales with the product.

    The search grid decides how far off the seed handed to tracking can be:

        code phase, worst case = +/- 0.5 sample / code_phase_factor
        Doppler,    worst case = +/- 0.5 * fft_resolution / doppler_factor

    Both are reported by `search_resolution_summary()`.
    """

    coherent_duration_replica_ms: int
    num_blocks: int
    sample_rate: float
    min_search_doppler_hz: float
    max_search_doppler_hz: float
    # None means "one coherent integration per replica period", the behaviour
    # every existing caller relies on.
    coherent_duration_sample_ms: float | None = None
    # (code_phase_factor, doppler_factor); None means (1, 1).
    fine_search_factors: tuple[int, int] | None = None

    def __post_init__(self):
        self.replica_length_samples = int(
            self.sample_rate * self.coherent_duration_replica_ms / 1000
        )
        self.replica_time_arr = (
            np.arange(self.replica_length_samples) / self.sample_rate
        )

        if self.coherent_duration_sample_ms is None:
            self.coherent_duration_sample_ms = float(self.coherent_duration_replica_ms)
        if not 0 < self.coherent_duration_sample_ms <= self.coherent_duration_replica_ms:
            raise ValueError(
                f"coherent_duration_sample_ms must satisfy 0 < coherent <= replica "
                f"({self.coherent_duration_replica_ms} ms), got {self.coherent_duration_sample_ms}"
            )
        self.coherent_length_samples = int(
            self.sample_rate * self.coherent_duration_sample_ms / 1000
        )
        if self.coherent_length_samples < 1:
            raise ValueError(
                f"coherent_duration_sample_ms {self.coherent_duration_sample_ms} is shorter than one "
                f"sample period at {self.sample_rate} Hz"
            )

        # Grid spacing comes from the FFT length; mainlobe width from how much
        # data is actually integrated.  They coincide only when the two lengths do.
        self.fft_resolution = self.sample_rate / self.replica_length_samples
        self.doppler_response_width_hz = self.sample_rate / self.coherent_length_samples

        assert self.num_blocks > 0
        self.acq_total_duration_ms = self.num_blocks * self.coherent_duration_sample_ms
        self.total_num_samples = int(
            self.acq_total_duration_ms / 1000 * self.sample_rate
        )
        self.min_doppler_fft_bin = int(self.min_search_doppler_hz / self.fft_resolution)
        self.max_doppler_fft_bin = int(self.max_search_doppler_hz / self.fft_resolution)
        self.doppler_search_bins = np.arange(
            self.min_doppler_fft_bin, self.max_doppler_fft_bin
        )
        self.num_doppler_bins = len(self.doppler_search_bins)

        # --- fine search -----------------------------------------------------
        code_phase_factor, doppler_factor = self.fine_search_factors or (1, 1)
        if code_phase_factor < 1 or doppler_factor < 1:
            raise ValueError(
                f"fine_search_factors must be >= 1, got {self.fine_search_factors}"
            )
        if code_phase_factor != 1:
            # The Doppler side is a re-mix of the same samples, so sub-dividing it
            # is the search that already exists, run at more offsets.  Code phase
            # is not: the bin *is* the sample spacing, so sub-dividing it needs
            # either interpolation of the correlation peak or a resampled replica.
            raise NotImplementedError(
                f"code-phase fine search (factor {code_phase_factor}) is not implemented yet; "
                "use fine_search_factors=(1, N) for now"
            )
        self.code_phase_factor = code_phase_factor
        self.doppler_factor = doppler_factor

        # Worst-case seeding error: half a bin, divided by the sub-division.
        self.code_phase_bin_seconds = 1.0 / self.sample_rate
        self.code_phase_error_seconds = 0.5 * self.code_phase_bin_seconds / code_phase_factor
        self.doppler_error_hz = 0.5 * self.fft_resolution / doppler_factor

        self.replica_cache_dict: Dict[str, SignalReplicaCacheEntry] = {}

    def search_resolution_summary(self) -> str:
        """
        Human-readable statement of how tightly the search pins each unknown --
        i.e. the worst-case error in the seed handed to tracking.  The code-phase
        error is also given in metres, since one sample of code phase is a range
        error of that size.
        """
        speed_of_light_m_per_s = 299792458.0
        coarse_code_ns = 0.5 * self.code_phase_bin_seconds * 1e9
        coarse_dopp_hz = 0.5 * self.fft_resolution
        lines = [
            f"Search grid: code phase {self.code_phase_bin_seconds * 1e9:.1f} ns/bin "
            f"({self.sample_rate / 1e6:.1f} Msps), Doppler {self.fft_resolution:.1f} Hz/bin",
            f"  plain search    -> code phase +/-{coarse_code_ns:.1f} ns "
            f"(+/-{coarse_code_ns * 1e-9 * speed_of_light_m_per_s:.1f} m), "
            f"Doppler +/-{coarse_dopp_hz:.1f} Hz",
        ]
        if self.fine_search_factors is not None:
            fine_code_ns = self.code_phase_error_seconds * 1e9
            lines.append(
                f"  fine search {self.fine_search_factors} -> code phase +/-{fine_code_ns:.1f} ns "
                f"(+/-{fine_code_ns * 1e-9 * speed_of_light_m_per_s:.1f} m), "
                f"Doppler +/-{self.doppler_error_hz:.1f} Hz"
            )
        return "\n".join(lines)


@dataclass
class AcqSignalCodeParameters:
    rate_chips_per_sec: float
    length_chips: int
    # Stated on the signal's own chip axis, carrying 0 wherever this component
    # does not transmit -- so a chip-interleaved component (e.g. L2C's CM) is
    # already zero-filled on its sibling's chips and cannot spuriously correlate
    # against that sibling's code. See `utils.code_components`.
    sequence: NDArray[np.int8]


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
    # The total the caller asked for, across this signal's whole search grid, and
    # the per-cell rate it was converted into.  Both are kept because only the
    # first is meaningful to a human and only the second sets the threshold.
    prob_false_alarm_total: float
    prob_false_alarm_per_cell: float
    num_detection_cells: int
    detection_threshold: float
    noise_var: float
    signal_detected: bool
    correlation_result: CorrelationResult
    config: AcquisitionConfiguration
    # Sub-bin refinement, in Hz, from the half-bin Doppler search.  Zero when the
    # peak came from the plain integer-bin pass (and for all signals when the
    # search is disabled).
    doppler_offset_hz: float = 0.0
    # The recovered code phase is only pinned modulo the period of the code that
    # was correlated against, because that code repeats within the replica.  For
    # L5 acquired on Q x NH20 that is 20 ms; for L1 C/A it is 1 ms.
    code_phase_ambiguity_ms: float = 0.0

    @property
    def peak_snr_db(self) -> float:
        """
        Peak height in dB above the *expected noise level*.

        `normalized_peak_value` is the peak divided by the noise variance, which
        under noise alone is chi-squared with `2M` degrees of freedom and so has
        mean `2M`.  Dividing that out puts 0 dB at the noise floor, which is the
        number a person can reason about; the raw ratio moves with `num_blocks`
        even when nothing about the signal changed.
        """
        return float(10.0 * np.log10(self.normalized_peak_value / (2 * self.config.num_blocks)))

    @property
    def detection_threshold_db(self) -> float:
        """The detection threshold on the same dB-above-noise scale as `peak_snr_db`."""
        return float(10.0 * np.log10(self.detection_threshold / (2 * self.config.num_blocks)))

    @property
    def acq_code_phase_ms(self) -> float:
        """Recovered code phase in ms, modulo `code_phase_ambiguity_ms`."""
        return self.acq_code_phase_seconds * 1e3

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


def pack_coherent_blocks(
    sample_block: NDArray[np.complex64],
    num_blocks: int,
    fft_length_samples: int,
    coherent_length_samples: int,
) -> NDArray[np.complex64]:
    """
    Lay each coherent block into a full-length FFT window, zero elsewhere.

    A block shorter than the replica has to be zero-padded, and *where* the data
    sits in that window matters.  Block `j` was received at samples
    `[j * coherent_length, ...)`, so the code had already advanced that far; its
    correlation peak therefore sits at lag `p + j * coherent_length`.  Padding
    every block at position 0 would leave each peak at a different lag and the
    square-law sum would smear them across the code phase axis instead of
    stacking them.

    Placing block `j` at the offset it actually occupied within the code period --
    `j * coherent_length` modulo the period -- puts every peak back at the same
    lag `p`.  Equivalently: each block is the whole dwell window with everything
    outside its own slice zeroed, so all blocks share one time origin.

    Zero-padding the *data* rather than the replica is what keeps the circular
    correlation valid: the replica stays a whole code period, so its wraparound
    is the code's own periodic extension, while the zeros contribute nothing.

    When `coherent_length == fft_length` every offset is zero and this reduces
    exactly to the reshape it replaces.
    """
    required = num_blocks * coherent_length_samples
    if len(sample_block) < required:
        raise ValueError(
            f"need {required} samples for {num_blocks} x {coherent_length_samples}-sample "
            f"coherent blocks, got {len(sample_block)}"
        )

    blocks = np.zeros((num_blocks, fft_length_samples), dtype=sample_block.dtype)
    for j in range(num_blocks):
        src = sample_block[j * coherent_length_samples : (j + 1) * coherent_length_samples]
        start = (j * coherent_length_samples) % fft_length_samples
        stop = start + coherent_length_samples
        if stop <= fft_length_samples:
            blocks[j, start:stop] = src
        else:
            # The block straddles a code period boundary; it wraps, and the
            # circular correlation handles the wrap correctly.
            split = fft_length_samples - start
            blocks[j, start:] = src[:split]
            blocks[j, : stop - fft_length_samples] = src[split:]
    return blocks


def run_acquisition(
    sample_block: NDArray[np.complex64],
    sample_block_uptime_epoch_ms: float,
    acq_config: AcquisitionConfiguration,
    code_parameters: Dict[str, AcqSignalCodeParameters],
    prob_false_alarm_total: float,
    print_progress: bool = False,
    noise_var_method: str = "abscorrmean",
    save_corr_peak_window_chips: Optional[float] = None,
) -> Dict[str, AcquisitionResult]:
    """
    Perform BPSK acquisition on the given sample block for all signals defined in code_parameters.

    Returns a dictionary mapping signal IDs to their respective AcquisitionResult.

    `prob_false_alarm_total` is the chance that *any* cell in one signal's search
    grid false-alarms -- a number that keeps its meaning when the sample rate,
    replica length or Doppler range change.  It is converted to a per-cell rate
    here; callers should not pre-correct it.  Across a sweep of P satellites,
    expect about `P * prob_false_alarm_total` false acquisitions, so this is the
    knob for trading sensitivity against false alarms:

        1e-3   sensitive: ~0.03 false acquisitions per 32-PRN sweep
        1e-6   balanced (about 1 dB stricter than 1e-3 on a large grid)
        1e-9   conservative; the extra ~0.7 dB buys little, because what actually
               produces false acquisitions in GNSS is cross-correlation against a
               strong satellite, which no threshold setting addresses.

    Pre-computed replicas and FFTs are stored for each signal in the config `replica_cache_dict`.
    The values in the cache entries must be recomputed when acquisition parameters change or when the sampling rate changes.

    Options for noise variance estimation:
        "abscorrmean": compute noise variance from the mean of abs(corr)**2 matrix
        "abscorrvar": compute noise variance from the square root of the variance of abs(corr)**2 matrix

    Sub-bin Doppler refinement is configured by `acq_config.fine_search_factors`.
    A Doppler factor of `k` repeats the search with the samples shifted down by
    `j/k` of a bin for each `j`, keeping whichever pass peaks highest -- so the
    worst-case Doppler error falls to `1/k` of half a bin, at `k` times the cost.
    It matters whenever the seed would otherwise be looser than the tracking loop
    can pull in from.
    """
    acquisition_results: Dict[str, AcquisitionResult] = {}

    #
    # Lay M coherent blocks into full-length FFT windows.  N is the replica (and
    # FFT) length; N_coh is how much data each block actually integrates, which is
    # shorter whenever the caller has capped coherent integration.
    M = acq_config.num_blocks
    N = acq_config.replica_length_samples
    N_coh = acq_config.coherent_length_samples
    samples = pack_coherent_blocks(sample_block, M, N, N_coh)

    # Convert the caller's total into the per-cell rate the threshold needs.
    #
    # This is deliberately not the caller's job.  The right per-cell value depends
    # on sample rate, replica duration and Doppler range, so a hard-coded one
    # silently stops meaning what it did the moment any of those change: 1e-7 per
    # cell is 0.025 expected false alarms over a 250,000-cell grid and 2.5 over a
    # 25,000,000-cell one.  The total is the quantity a person can reason about.
    #
    # Sidak, in the numerically stable form: 1 - (1 - p)**(1/n) cancels to exactly
    # 0.0 once p/n drops below float64 eps, and chi2.isf(0) is inf, so the naive
    # expression fails by detecting nothing at all rather than by erroring.
    num_detection_cells = acq_config.num_doppler_bins * acq_config.replica_length_samples
    # Each sub-bin pass is another set of Doppler hypotheses, and every hypothesis
    # is another chance to false-alarm, so the grid the threshold is set for grows
    # with the factor.
    num_detection_cells *= acq_config.doppler_factor
    if not 0.0 < prob_false_alarm_total < 1.0:
        raise ValueError(
            f"prob_false_alarm_total must be in (0, 1), got {prob_false_alarm_total}"
        )
    prob_false_alarm_per_cell = -np.expm1(
        np.log1p(-prob_false_alarm_total) / num_detection_cells
    )
    # The threshold depends only on the per-cell rate and the number of blocks, so
    # it is one number for the whole sweep rather than a per-signal quantity.
    detection_threshold = float(scipy.stats.chi2(df=2 * M).isf(prob_false_alarm_per_cell))
    threshold_db = 10.0 * np.log10(detection_threshold / (2 * M))
    # Compute conjugated FFT of blocks, once per sub-bin Doppler hypothesis.
    # Blocks are combined non-coherently below, so applying the same phase ramp
    # from the start of each block (rather than from the start of the capture) is
    # sufficient -- only the frequency shift within a block matters.  The ramp is
    # applied across the padded window; the padding stays zero, and the block's
    # own position within the window contributes only a constant phase, which the
    # square law discards.
    sub_bin_offsets_hz = [0.0]
    conj_sample_ffts = [np.conj(np.fft.fft(samples, axis=1))]
    block_time_arr = np.arange(N) / acq_config.sample_rate
    for j in range(1, acq_config.doppler_factor):
        # Mixing the samples down by j/k of a bin makes the signal appear that much
        # lower, so the recovered Doppler is the bin centre plus the offset.
        offset_hz = j * acq_config.fft_resolution / acq_config.doppler_factor
        shifted_samples = samples * np.exp(-2j * np.pi * offset_hz * block_time_arr)[None, :]
        conj_sample_ffts.append(np.conj(np.fft.fft(shifted_samples, axis=1)))
        sub_bin_offsets_hz.append(offset_hz)

    if print_progress:
        # The code period is what code phase is ambiguous over; state it in the
        # header when every signal shares one, rather than on every row.
        code_periods_ms = {
            1e3 * p.length_chips / p.rate_chips_per_sec for p in code_parameters.values()
        }
        ambiguity = (
            f", mod {code_periods_ms.pop():g}" if len(code_periods_ms) == 1 else ""
        )
        print(
            f"Detection threshold: {threshold_db:.2f} dB above noise "
            f"(p_fa {prob_false_alarm_total:g} over {num_detection_cells:,} cells/PRN)"
        )
        print()
        print(f"  {'PRN':<5} {'SNR [dB]':>9} {'Doppler [Hz]':>13} {'Code phase [ms' + ambiguity + ']':>22}")
        print(f"  {'-' * 5} {'-' * 9} {'-' * 13} {'-' * 22}")

    for signal_id, code_params in code_parameters.items():

        # The correlation is circular, so the replica's wraparound is only the
        # code's own periodic extension if the replica spans a whole number of code
        # periods.  Half a period aliases silently: a true code phase in the
        # uncovered part is reported inside the covered part, with a healthy peak
        # and signal_detected set.  Cheap to check, and it is not otherwise
        # checkable anywhere -- the config does not know the code, and the code
        # parameters do not know the replica length.
        code_period_ms = code_params.length_chips / code_params.rate_chips_per_sec * 1e3
        periods_per_replica = acq_config.coherent_duration_replica_ms / code_period_ms
        if abs(periods_per_replica - round(periods_per_replica)) > 1e-9 or periods_per_replica < 1:
            raise ValueError(
                f"{signal_id}: coherent_duration_replica_ms {acq_config.coherent_duration_replica_ms} must be a "
                f"positive whole number of code periods ({code_period_ms:g} ms for "
                f"{code_params.length_chips} chips at {code_params.rate_chips_per_sec:g} cps), "
                f"got {periods_per_replica:g}; a partial period aliases the code phase"
            )

        # Check if there is a cached replica for this signal
        if signal_id in acq_config.replica_cache_dict:
            replica_entry = acq_config.replica_cache_dict[signal_id]
            replica_samples_fft = replica_entry.replica_fft
        else:
            replica_samples = np.zeros(
                N, dtype=np.complex64
            )
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
        # With a Doppler factor of 1 there is exactly one hypothesis and this
        # reduces to the plain single pass.
        correlation = None
        peak_doppler_bin = peak_sample_bin = 0
        peak_val = -np.inf
        doppler_offset_hz = 0.0

        for conj_samples_fft, sub_bin_offset_hz in zip(conj_sample_ffts, sub_bin_offsets_hz):
            pass_correlation = np.zeros((len(doppler_search_bins), N))

            for i, roll in enumerate(doppler_search_bins):
                # Coherent integration over the N_coh non-zero samples of each
                # block; the padding contributes nothing, so z_noise ~ CN(0,
                # N_coh*noise_var) regardless of the FFT length.
                shifted_replica_fft = np.roll(replica_samples_fft, roll)
                corr = np.fft.ifft(
                    conj_samples_fft * shifted_replica_fft[None, :]
                )
                # non-coherent square-law summation over M blocks, normalized by
                # the number of samples actually integrated (N_coh, not N -- they
                # differ once coherent integration is shorter than the replica, and
                # normalizing by N would under-report the noise by N/N_coh)
                # y_noise / noise_var ~ ChiSquared(2M)
                pass_correlation[i] = np.sum(1 / N_coh * np.abs(corr) ** 2, axis=0)

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
            prob_false_alarm_total,
            prob_false_alarm_per_cell,
            num_detection_cells,
            detection_threshold,
            noise_var,
            signal_detected,
            corr_result,
            acq_config,
            doppler_offset_hz=doppler_offset_hz,
            code_phase_ambiguity_ms=code_period_ms,
        )

        acquisition_results[signal_id] = acq_result

        if print_progress:
            # Doppler and code phase are only meaningful where a peak was actually
            # detected; a dash is honest about the rest being the noise maximum.
            doppler = f"{acq_result.acq_doppler_hz:+.0f}" if signal_detected else "-"
            code_phase = f"{acq_result.acq_code_phase_ms:.3f}" if signal_detected else "-"
            print(
                f"  {signal_id:<5} {acq_result.peak_snr_db:9.1f} {doppler:>13} "
                f"{code_phase:>22}"
            )

    return acquisition_results
