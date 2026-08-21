from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from . import bpsk_acquisition, tracking_channel
from .collect_metadata_utils import ExperimentMetadata

if TYPE_CHECKING:  # avoids importing signal_interfaces at runtime
    from .signal_interfaces import TrackingChannelAdapter


def setup_default_plotting():
    """
    Sets up default plotting parameters for matplotlib.
    """
    plt.rcParams.update({
        'figure.dpi': 150,
        'figure.figsize': (10, 6),
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })


def plot_receiver_channel_bands(
        fig: Figure | SubFigure,
        metadata: ExperimentMetadata,
        include_bands: Optional[Iterable[str]] = None,
        exclude_bands: Optional[Iterable[str]] = None,
        samp_bandwidth_height: float = 0.5,
        samp_center_height: float = 0.8,
):
    axes = fig.subplots(1, 2, sharey=True)
    ax1: Axes = axes[0]
    ax2: Axes = axes[1]
    # Want to plot two axes;
    # Left shows baseband frequencies and right shows RF frequencies
    # Channel IDs go down the y-axis
    band_colors = {}
    for i, band_id in enumerate(metadata.band_ids):
        band_colors[band_id] = f"C{i}"

    band_configurations = metadata.band_configurations
    for i, channel_id in enumerate(metadata.channel_ids):
        channel_config = metadata.channel_configurations[channel_id]
        samp_bandwidth_MHz = channel_config.samp_rate / 1e6
        is_real = not channel_config.sample_params.is_complex
        if is_real:
            # Only shade positive frequencies
            ax1.fill_betweenx([i - samp_bandwidth_height / 2, i + samp_bandwidth_height / 2], 0, samp_bandwidth_MHz / 2, color="r", alpha=0.3)
        else:
            ax1.fill_betweenx([i - samp_bandwidth_height / 2, i + samp_bandwidth_height / 2], -samp_bandwidth_MHz / 2, samp_bandwidth_MHz / 2, color="b", alpha=0.3)

        for band_id in channel_config.band_ids:
            band_config = band_configurations[band_id]
            baseband_if_MHz = band_config.inter_freq / 1e6
            rf_center_MHz = band_config.center_freq / 1e6

            # Plot delta-like markers at IF (baseband) and RF center frequencies.
            ax1.vlines(baseband_if_MHz, i - samp_center_height / 2, i + samp_center_height / 2, color="k", linewidth=2)
            # ax1.plot([baseband_if], [i], marker="|", markersize=12, color="C0")

            ax2.vlines(rf_center_MHz, i - samp_center_height / 2, i + samp_center_height / 2, color="k", linewidth=2)
            # ax2.plot([rf_center], [i], marker="|", markersize=12, color="C1")

            rf_band_center_MHz = (band_config.center_freq - band_config.inter_freq) / 1e6
            if is_real:
                ax2.fill_betweenx([i - samp_bandwidth_height / 2, i + samp_bandwidth_height / 2], rf_band_center_MHz, rf_band_center_MHz + samp_bandwidth_MHz / 2, color="r", alpha=0.3)
            else:
                ax2.fill_betweenx([i - samp_bandwidth_height / 2, i + samp_bandwidth_height / 2], rf_band_center_MHz - samp_bandwidth_MHz / 2, rf_band_center_MHz + samp_bandwidth_MHz / 2, color="r", alpha=0.3)

    ax1.set_yticks(range(len(metadata.channel_ids)))
    ax1.set_yticklabels(metadata.channel_ids)
    for ax in [ax1, ax2]:
        ax.grid()
    ax1.set_xlabel("Baseband Frequency [MHz]")
    ax2.set_xlabel("RF Frequency [MHz]")


# --- raw sample diagnostics -------------------------------------------------

def plot_raw_sample_histogram(fig: Figure | SubFigure, sample_buffer: np.ndarray, hist_bins: Optional[np.ndarray] = None) -> Axes:
    """Histogram of a raw complex sample buffer's real/imaginary components."""
    ax = fig.add_subplot(1, 1, 1)
    if hist_bins is None:
        hist_bins = np.arange(-128, 128)
    ax.hist(sample_buffer.real, bins=hist_bins, rwidth=0.5, color="r", align="left", label="Real")
    ax.hist(sample_buffer.imag, bins=hist_bins, rwidth=0.5, color="b", align="mid", label="Imaginary")
    ax.set_title("Histogram of Raw Samples")
    ax.set_xlabel("Sample Value")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid()
    return ax


def plot_welch_psd(
    fig: Figure | SubFigure,
    baseband_samples: np.ndarray,
    samp_rate: float,
    orig_samples: np.ndarray | None = None,
    nperseg: int = 4096,
    noverlap: int = 2048,
) -> Axes:
    """Welch PSD estimate of a raw sample buffer overlaid with its mixed-down baseband."""
    ax = fig.add_subplot(1, 1, 1)
    freqs, psd = scipy.signal.welch(
        baseband_samples, fs=samp_rate, nperseg=nperseg, noverlap=noverlap,
        window="hann", return_onesided=False, scaling="density",
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    if orig_samples is not None:
        _, psd_orig = scipy.signal.welch(
            orig_samples, fs=samp_rate, nperseg=nperseg, noverlap=noverlap,
            window="hann", return_onesided=False, scaling="density",
        )
        psd_orig = np.fft.fftshift(psd_orig)
        ax.plot(freqs / 1e6, 10 * np.log10(psd_orig), color="gray", label="Original Samples")

    ax.plot(freqs / 1e6, 10 * np.log10(psd), color="black", label="Baseband Samples")
    
    ax.set_title("Welch PSD Estimate of Raw Samples")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    ax.grid()
    ax.legend()
    return ax


def plot_sample_histogram_and_constellation(
    fig: Figure | SubFigure,
    samples: np.ndarray,
    bit_depth: int = 8,
) -> Sequence[Axes]:
    """
    Two views of one buffer of raw samples: a histogram of the I and Q values,
    and the I-vs-Q scatter ("constellation").

    What a healthy collect looks like: both histograms are bell-shaped (Gaussian)
    and centred near zero, and the scatter is a round, featureless blob. GNSS
    signals arrive far below the noise floor, so what you are looking at is
    essentially receiver noise -- the satellites are invisible until correlation
    pulls them out.

    What problems look like:
      - Histogram pressed flat against the ends of the range -> the front-end gain
        is too high and samples are clipping.
      - Histogram squeezed into just a few values near zero -> gain too low, and
        quantisation is throwing away the signal.
      - Q identically zero -> the data is real-valued, not complex; check
        `is_complex` in the collect's metadata.yml.
      - An off-centre blob, or a ring/arc rather than a disc -> a DC bias or an
        uncorrected carrier offset.

    `bit_depth` sets the histogram range to the full span the sample format can
    represent, so an under-driven collect is obvious by how little of the axis it
    fills.
    """
    axes = fig.subplots(1, 2, width_ratios=[1.5, 1])
    ax_hist: Axes = axes[0]
    ax_scatter: Axes = axes[1]

    hist_bins = np.arange(-(2 ** (bit_depth - 1)), 2 ** (bit_depth - 1))
    ax_hist.hist(samples.real, bins=hist_bins, histtype="stepfilled", color="r", alpha=0.6, align="left", label="Real (I)")
    ax_hist.hist(samples.imag, bins=hist_bins, histtype="stepfilled", color="b", alpha=0.6, align="mid", label="Imaginary (Q)")
    ax_hist.set_xlabel("Sample Value")
    ax_hist.set_ylabel("Count")
    ax_hist.grid()
    ax_hist.legend(loc="upper right")

    # alpha is very low because a 40 ms buffer is ~1e6 points: the density, not
    # any single dot, is the thing to read.
    ax_scatter.scatter(samples.real, samples.imag, color="k", s=1, alpha=0.01, zorder=1)
    ax_scatter.set_axisbelow(True)
    ax_scatter.grid()
    ax_scatter.set_xlabel("Real (I)")
    ax_scatter.set_ylabel("Imaginary (Q)")
    return axes


def plot_stft_periodogram(
    fig: Figure | SubFigure,
    periodogram: np.ndarray,
    samp_rate: float,
    total_duration_s: float,
) -> Axes:
    """
    Spectrogram: how the power spectrum of the collect changes over time.

    `periodogram` is (num_windows, num_freq_bins), one PSD estimate per time
    window, already fftshifted so frequency runs monotonically from -samp_rate/2
    to +samp_rate/2. Colour is power in dB.

    What to look for: a steady horizontal band across the whole capture means the
    front end behaved consistently. Vertical stripes are momentary interference or
    dropped samples; a band that brightens or fades over time means the gain (or
    the antenna's view of the sky) changed mid-collect. Narrow horizontal lines
    that persist are continuous-wave interference -- a jammer or a nearby
    oscillator -- which is exactly the kind of thing that stops acquisition from
    working later.
    """
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        10 * np.log10(periodogram.T),
        aspect="auto",
        origin="lower",
        extent=[0, total_duration_s, -samp_rate / 2e6, samp_rate / 2e6],
        interpolation="nearest",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [MHz]")
    ax.set_title("STFT Periodogram")
    fig.colorbar(im, ax=ax, label="Power/Frequency [dB/Hz]")
    return ax


# --- acquisition diagnostics -------------------------------------------------

def plot_acquisition_peak_values(
    fig: Figure | SubFigure,
    acq_results: dict[str, bpsk_acquisition.AcquisitionResult],
) -> Axes:
    """Stem plot of normalized peak correlation value per signal, with the detection threshold."""
    ax = fig.add_subplot(1, 1, 1)
    all_sig_ids = sorted(acq_results.keys())
    all_peak_vals = [acq_results[sig_id].normalized_peak_value for sig_id in all_sig_ids]
    ax.stem(range(len(all_peak_vals)), all_peak_vals, basefmt=" ")
    detection_threshold = acq_results[all_sig_ids[0]].detection_threshold
    ax.plot([0, len(all_sig_ids)], [detection_threshold] * 2, color="red", linestyle="--", label="Detection Threshold")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xticks(range(len(all_sig_ids)))
    ax.set_xticklabels(all_sig_ids, rotation=45)
    ax.set_title("Acquisition Peak Correlation Values")
    ax.set_xlabel("Signal ID (PRN)")
    ax.set_ylabel("Normalized Peak Correlation Value")
    ax.grid()
    return ax


def plot_acquisition_doppler_slices(
    fig: Figure | SubFigure,
    acq_results: dict[str, bpsk_acquisition.AcquisitionResult],
) -> Axes:
    """Correlation-vs-Doppler slice through each signal's peak code-phase bin."""
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap("tab20b")
    for i, (signal_id, acq_result) in enumerate(acq_results.items()):
        color = cmap(i / 40.0)
        marker = ("o", "x")[i % 2]
        ax.plot(
            acq_result.correlation_result.doppler_bins_hz,
            acq_result.correlation_result.correlation_matrix[:, acq_result.peak_code_phase_bin],
            color=color, marker=marker, label=f"{signal_id}",
        )
    ax.set_yscale("log")
    ax.set_title("Acquisition Correlation Results")
    ax.set_xlabel("Doppler Frequency [Hz]")
    ax.set_ylabel("Peak Slice Correlation Magnitude")
    ax.legend(ncol=4, fontsize=8)
    ax.grid()
    return ax


def plot_acquisition_delay_doppler_map(
    fig: Figure | SubFigure,
    acq_result: bpsk_acquisition.AcquisitionResult,
    code_phase_window_samples: int = 150,
) -> Axes:
    """Delay-Doppler correlation heatmap for one signal's acquisition result, zoomed to the peak."""
    ax = fig.add_subplot(1, 1, 1)
    acq_config = acq_result.config
    correlation = acq_result.correlation_result.correlation_matrix
    num_doppler_bins, num_code_phases = correlation.shape
    extent = [0, num_code_phases, acq_config.min_search_doppler_hz, acq_config.max_search_doppler_hz]

    peak_code_phase_bin = acq_result.peak_code_phase_bin
    im = ax.imshow(
        correlation, extent=extent, aspect="auto", interpolation="nearest",
        cmap="plasma", origin="lower", vmin=0,
    )
    ax.set_xlim(peak_code_phase_bin - code_phase_window_samples, peak_code_phase_bin + code_phase_window_samples)
    ax.set_xlabel("Code Phase [samples]")
    ax.set_ylabel("Doppler Frequency [Hz]")
    ax.set_title(f"Delay-Doppler Correlation Map for {acq_result.signal_id}")
    fig.colorbar(im, ax=ax, label="Correlation Magnitude")
    return ax


def plot_acquisition_correlation_histogram(
    fig: Figure | SubFigure,
    acq_result: bpsk_acquisition.AcquisitionResult,
    num_blocks: int,
    hist_max_val: float = 120.0,
) -> Axes:
    """
    Histogram of normalized correlation magnitudes for one signal, with the
    chi-squared distribution (df = 2 * num_blocks) that non-coherent
    square-law summation should follow under noise alone, and the detection
    threshold used to declare acquisition.
    """
    ax = fig.add_subplot(1, 1, 1)
    corr_matrix = acq_result.correlation_result.correlation_matrix

    y_noise_mean = np.mean(corr_matrix)
    sigma_n = np.sqrt(y_noise_mean / (2 * num_blocks))
    normalized_corr_matrix = corr_matrix / sigma_n**2

    hist_bins = np.linspace(0, hist_max_val, 100)
    hist = np.histogram(normalized_corr_matrix.flatten(), bins=hist_bins)[0]

    ax.bar(hist_bins[:-1], hist, width=hist_bins[1] - hist_bins[0], color="blue", alpha=0.7)
    x_vals = np.linspace(0, np.max(normalized_corr_matrix), 1000)
    chi2_pdf = scipy.stats.chi2.pdf(x_vals, df=2 * num_blocks)
    ax.plot(
        x_vals, chi2_pdf * np.max(hist) / np.max(chi2_pdf), color="red", linewidth=2,
        label=f"Chi-squared PDF (df={2 * num_blocks})",
    )
    ax.vlines([acq_result.detection_threshold], ymin=0, ymax=np.max(hist), color="black", linestyle="--", label="Detection Threshold")
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(1, 1e6)
    ax.set_title(f"Histogram of Correlation Magnitudes for {acq_result.signal_id}")
    ax.set_xlabel("Correlation Magnitude")
    ax.set_ylabel("Count")
    ax.grid()
    ax.set_xlim(0, hist_max_val)
    return ax


def plot_acquisition_dwell_layout(
    fig: Figure | SubFigure,
    acq_config: bpsk_acquisition.AcquisitionConfiguration,
    symbol_period_ms: Optional[float] = None,
    symbol_phase_ms: float = 0.0,
) -> Sequence[Axes]:
    """
    Why acquisition has two lengths, drawn from the configuration itself.

    Left panel -- the dwell in time.  `num_blocks` coherent blocks of
    `coherent_duration_sample_ms` are taken back to back from the stream, and each
    is zero-padded out to `coherent_duration_replica_ms` before its FFT (the
    correlation is circular, so the FFT length is the replica length).

    Each block sits at the offset it actually occupied within the code period --
    block `j` at `j * T_coherent` modulo `T_replica`, wrapping if it straddles the
    end -- not at the start of its window.  That is what `pack_coherent_blocks`
    does, and it is what makes the square-law sum accumulate: block `j` was
    received after the code had already advanced that far, so padding every block
    at position 0 would leave each peak at a different lag and smear the sum
    across the code phase axis instead of stacking it.

    If `symbol_period_ms` is given, data symbol boundaries are drawn across the
    window.  A sign flip *between* blocks is harmless because the blocks are
    combined by square law; one *inside* a block cancels part of that block's own
    integration.  `symbol_phase_ms` offsets that grid: acquisition does not know
    the symbol alignment, so the realistic picture is an arbitrary offset, and
    keeping blocks short is what bounds the damage whatever it turns out to be.

    Right panel -- the same two lengths in frequency.  Grid ticks are the Doppler
    bins the FFT actually produces, spaced `1 / T_replica`.  The curve is the
    coherent response, whose mainlobe is `1 / T_coherent` wide.  A short coherent
    length therefore widens the response without widening the grid, which is what
    lets a short integration still be located to a fine Doppler.
    """
    axes = fig.subplots(1, 2, width_ratios=[1.4, 1])
    ax_time: Axes = axes[0]
    ax_freq: Axes = axes[1]

    t_coh = float(acq_config.coherent_duration_sample_ms)
    t_rep = float(acq_config.coherent_duration_replica_ms)
    num_blocks = acq_config.num_blocks

    for m in range(num_blocks):
        y = num_blocks - 1 - m
        # Where this block's data actually sits in the window, wrapping if it
        # straddles the end of the code period.
        start = (m * t_coh) % t_rep
        spans = [(start, min(t_coh, t_rep - start))]
        if start + t_coh > t_rep:
            spans.append((0.0, start + t_coh - t_rep))

        ax_time.broken_barh([(0.0, t_rep)], (y - 0.32, 0.64),
                            facecolors="lightgrey", edgecolor="k",
                            linewidth=0.5, hatch="//")
        ax_time.broken_barh(spans, (y - 0.32, 0.64),
                            facecolors="tab:blue", edgecolor="k", linewidth=0.5)
        ax_time.text(spans[0][0] + spans[0][1] / 2, y, f"{m}", ha="center",
                     va="center", fontsize=8, color="white")

    if symbol_period_ms:
        # Boundaries are absolute in the window: it is one code period, and the
        # blocks have been placed back onto their true positions within it.
        edges = np.arange(symbol_phase_ms % symbol_period_ms, t_rep, symbol_period_ms)
        ax_time.vlines(edges, -0.5, num_blocks - 0.5, color="tab:red", lw=2, zorder=3)

    ax_time.set_yticks(range(num_blocks))
    ax_time.set_yticklabels([f"{num_blocks - 1 - i}" for i in range(num_blocks)])
    ax_time.set_ylabel("Block")
    ax_time.set_xlabel("Time within the FFT window [ms]")
    ax_time.set_xlim(0, t_rep)
    ax_time.set_title(f"Dwell: {num_blocks} x {t_coh:g} ms coherent, {t_rep:g} ms replica")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="tab:blue", edgecolor="k", label="data integrated"),
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgrey", edgecolor="k", hatch="//", label="zero padding"),
    ]
    if symbol_period_ms:
        handles.append(plt.Line2D([0], [0], color="tab:red", lw=2,
                                  label=f"symbol boundary ({symbol_period_ms:g} ms)"))
    ax_time.legend(handles=handles, fontsize=7, loc="upper right")

    # --- frequency ---
    grid_hz = acq_config.fft_resolution
    response_hz = acq_config.doppler_response_width_hz
    span = 3 * response_hz
    f = np.linspace(-span, span, 1001)
    ax_freq.plot(f, np.abs(np.sinc(f / response_hz)), color="k", lw=2,
                 label=f"response, {response_hz:.0f} Hz wide")
    ticks = np.arange(-span, span + grid_hz, grid_hz)
    ax_freq.vlines(ticks, 0, 0.12, color="tab:orange", lw=1.5,
                   label=f"Doppler bins, {grid_hz:.0f} Hz apart")
    ax_freq.set_xlabel("Doppler offset from the true value [Hz]")
    ax_freq.set_ylabel("Normalised correlation")
    ax_freq.set_title("Grid spacing vs response width")
    ax_freq.set_xlim(-span, span)
    ax_freq.set_ylim(0, 1.1)
    ax_freq.grid()
    ax_freq.legend(fontsize=7, loc="upper right")
    return axes


# --- tracking-result diagnostics --------------------------------------------

def corr_component(corr: np.ndarray, index: int = 0) -> np.ndarray:
    """
    Pick one code component out of a correlator output array.

    Correlator outputs are (epochs, components) so that multi-component signals
    such as L2C/L5 keep their components separate. Results pickled before that
    change are 1-D, so they are passed through unchanged.
    """
    return corr[:, index] if np.ndim(corr) == 2 else corr


def plot_prompt_iq_grid(
    fig: Figure | SubFigure,
    tracking_outputs: dict[str, tracking_channel.SignalTrackingOutputs],
    sig_ids: Optional[Sequence[str]] = None,
    title: str = "",
) -> Sequence[Axes]:
    """One row per signal of prompt I (red) / Q (blue) scatter vs. uptime."""
    if sig_ids is None:
        sig_ids = sorted(tracking_outputs.keys())
    axes = np.atleast_1d(fig.subplots(len(sig_ids), 1, sharex=True, sharey=True))
    for i, sig_id in enumerate(sig_ids):
        tracking_output = tracking_outputs[sig_id]
        valid = tracking_output.valid
        prompt = corr_component(tracking_output.prompt_corr)[valid]
        plot_time = tracking_output.uptime_epoch_ms[valid] * 1e-3

        ax: Axes = axes[i]
        ax.scatter(plot_time, prompt.real, s=1, color="tab:red")
        ax.scatter(plot_time, prompt.imag, s=1, color="tab:blue")
        ax.grid(True)
        ax.set_ylabel(f"{sig_id}", fontsize=22)

    if title:
        axes[0].set_title(title)
    axes[-1].set_xlabel("Uptime [seconds]")
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markersize=5, label='I'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=5, label='Q'),
    ]
    axes[0].legend(handles=handles, loc='upper right', markerscale=3)
    return axes


def plot_carrier_phase_doppler_comparison(
    fig: Figure | SubFigure,
    tracking_outputs_by_version: dict[str, dict[str, tracking_channel.SignalTrackingOutputs]],
    loop_params_by_version: dict[str, tracking_channel.TrackingLoopParameters],
    sig_id: str,
    version_ids: Optional[Sequence[str]] = None,
) -> Sequence[Axes]:
    """
    Compare carrier-phase error, Doppler, and detrended carrier phase for one
    signal across several tracking-loop parameter "versions", overlaid with a
    shared color scale (useful for e.g. sweeping PLL bandwidth).
    """
    if version_ids is None:
        version_ids = sorted(tracking_outputs_by_version.keys())
    axes = fig.subplots(3, 1, sharex=True)
    cmap = plt.get_cmap("viridis")

    for i, version_id in enumerate(version_ids[::-1]):
        loop_params = loop_params_by_version[version_id]
        tracking_output = tracking_outputs_by_version[version_id][sig_id]

        valid = tracking_output.valid
        plot_time = tracking_output.uptime_epoch_ms[valid][:-1] * 1e-3
        carr_phase_errors_cycles = tracking_output.carr_phase_errors_cycles[valid][:-1]
        doppler_freq_hz = tracking_output.doppler_freq_hz[valid][:-1]
        carr_phase_cycles = tracking_output.carr_phase_cycles[valid][:-1]

        ave_doppler_hz = np.mean(doppler_freq_hz)
        detr_carr_phase_cycles = carr_phase_cycles - ave_doppler_hz * plot_time
        detr_carr_phase_cycles -= detr_carr_phase_cycles[0]

        color = cmap(i / len(version_ids))

        axes[0].plot(plot_time, carr_phase_errors_cycles, color=color, lw=3)
        axes[1].plot(plot_time, doppler_freq_hz, color=color, lw=3)
        axes[2].plot(plot_time, detr_carr_phase_cycles, color=color, lw=3)

    axes[0].set_ylabel("Carrier Phase\nError [cycles]")
    axes[0].set_title(f"Tracking Results for Signal {sig_id}")
    axes[1].set_ylabel("Doppler\nFrequency [Hz]")
    axes[2].set_ylabel("Detrended Carrier\nPhase [cycles]")
    axes[2].set_xlabel("Uptime [seconds]")
    for ax in axes:
        ax.grid()

    handles = [
        plt.Line2D(
            [0], [0], color=cmap(i / len(version_ids)), lw=5,
            label=f"PLL BW: {loop_params_by_version[version_id].PLL_bandwidth_hz} Hz",
        )
        for i, version_id in enumerate(version_ids)
    ]
    axes[0].legend(handles=handles, loc="upper right", framealpha=1)
    fig.align_labels()
    return axes


def plot_epl_magnitude_and_code_error(
    fig: Figure | SubFigure,
    tracking_output: tracking_channel.SignalTrackingOutputs,
    sig_id: str,
) -> Sequence[Axes]:
    """Early/Prompt/Late correlation magnitude and code-phase error vs. uptime for one signal."""
    axes = fig.subplots(2, 1, sharex=True)

    valid = tracking_output.valid
    plot_time = tracking_output.uptime_epoch_ms[valid][:-1] * 1e-3
    early = corr_component(tracking_output.early_corr)[valid][:-1]
    prompt = corr_component(tracking_output.prompt_corr)[valid][:-1]
    late = corr_component(tracking_output.late_corr)[valid][:-1]
    code_phase_errors_chips = tracking_output.code_phase_errors_chips[valid][:-1]

    axes[0].scatter(plot_time, np.abs(early), color="g", s=5)
    axes[0].scatter(plot_time, np.abs(prompt), color="b", s=5)
    axes[0].scatter(plot_time, np.abs(late), color="r", s=5)
    axes[0].set_ylabel("EPL Magnitude")
    axes[0].set_title(f"EPL Magnitudes and Code Phase Error for Signal {sig_id}")
    axes[0].legend(["Early", "Prompt", "Late"], markerscale=10, loc="upper right")

    axes[1].plot(plot_time, code_phase_errors_chips, color=plt.get_cmap("viridis")(0.5), lw=3)
    axes[1].set_ylabel("Code Phase\nError [chips]")
    axes[1].set_xlabel("Uptime [seconds]")

    for ax in axes:
        ax.grid()
    fig.align_labels()
    return axes


def plot_component_prompt_magnitudes(
    fig: Figure | SubFigure,
    adapter: "TrackingChannelAdapter",
    sig_id: str,
    title: Optional[str] = None,
) -> Axes:
    """
    Prompt correlation magnitude of every code component of a multi-component
    signal, on one set of axes. Meaningless for a single-component signal such as
    L1 C/A, so callers should skip it when `len(component_names) == 1`.

    What to expect:
      - L5: I and Q carry equal power, so the two traces should sit on top of each
        other. A large gap means one component is not being tracked properly.
      - L2C: CM and CL each transmit on only half the chip slots, so both sit near
        half the magnitude a single full-rate code would reach -- and they should
        be roughly equal to each other.

    Magnitude is used rather than I/Q because it is insensitive to carrier phase:
    it answers "how much signal power is this component recovering", not "is the
    phase right".
    """
    ax = fig.add_subplot(1, 1, 1)
    outputs = adapter.outputs
    plot_time = outputs.uptime_epoch_ms[outputs.valid] * 1e-3
    for index, name in enumerate(adapter.signal.component_names):
        prompt = adapter.get_prompt_component(component=index)
        ax.scatter(plot_time, np.abs(prompt), s=2, label=f"{name}", color=f"C{index}")
    ax.set_title(title if title is not None else f"Components: {sig_id}")
    ax.set_ylabel("Prompt Magnitude")
    ax.set_xlabel("Uptime [s]")
    ax.grid(True)
    ax.legend(markerscale=10)
    return ax


def plot_prompt_components(
    fig: Figure | SubFigure,
    adapter: "TrackingChannelAdapter",
    sig_id: str,
    title: Optional[str] = None,
) -> Sequence[Axes]:
    """
    Prompt correlator output per code component, one row each, in-phase and
    quadrature together.

    A locked channel puts essentially all of each component's power on *one* axis
    and leaves the other at zero.  Which axis depends on the component's carrier
    phase relative to the one the loop is tracking:

      - the component the carrier loop runs on lands on I, because that is what the
        PLL is driving it to do;
      - a component transmitted in phase quadrature with it lands on Q.  GPS L5 is
        exactly this case -- I and Q ride quadrature carriers, and with the loop on
        the Q pilot the L5I component's power appears in the *imaginary* part;
      - components sharing a carrier phase (L2C's CM and CL) both land on I.

    How many bands that axis forms says what the component carries: a data
    component (L1 C/A's CA, L2C's CM, L5's I) splits into a positive and a negative
    band as navigation symbols flip its sign, while a dataless pilot (L5's Q, L2C's
    CL once resolved) stays in a single band.

    Loss of lock looks like I and Q both scattered symmetrically about zero.  Since
    every component shares one epoch, a component whose magnitude collapses on a
    subset of epochs while its siblings are healthy is straddling its own symbol
    boundary -- see `utils.tracking_channel`'s epoch anchoring.
    """
    outputs = adapter.outputs
    names = adapter.signal.component_names
    plot_time = outputs.uptime_epoch_ms[outputs.valid] * 1e-3

    axes = np.atleast_1d(fig.subplots(len(names), 1, sharex=True))
    for index, name in enumerate(names):
        prompt = adapter.get_prompt_component(component=index)
        ax: Axes = axes[index]
        ax.scatter(plot_time, prompt.real, s=2, color="tab:red", label="In-phase (I)")
        ax.scatter(plot_time, prompt.imag, s=2, color="tab:blue", label="Quadrature (Q)")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_ylabel(f"{name}\nPrompt")
        ax.grid(True)
    axes[0].set_title(title if title is not None else f"Prompt correlators: {sig_id}")
    axes[0].legend(markerscale=8, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Uptime [s]")
    fig.align_labels()
    return axes


def plot_code_delay_and_doppler(
    fig: Figure | SubFigure,
    adapter: "TrackingChannelAdapter",
    sig_id: str,
    title: Optional[str] = None,
) -> Axes:
    """
    Code delay and carrier Doppler on shared axes, as the line-of-sight dynamics.

    Code delay is plotted as the *residual*: the tracked code phase minus the
    nominal one-millisecond-per-millisecond advance, referenced to the first
    epoch.  The raw code phase is dominated by that nominal advance and shows
    nothing; the residual is the part that reflects the satellite actually moving,
    and is reported in chips (the right-hand axis carries Doppler in Hz).

    The two are related by construction, not independently measured: this tracker
    slaves the code rate to the carrier, `code_rate = (1 + doppler / f_carrier)`,
    so the delay residual is the integral of Doppler over the carrier frequency.
    What the plot is good for is seeing that dynamic directly -- a steady Doppler
    of a few kHz produces a delay ramp of a fraction of a chip per second -- and
    seeing both break together when a channel loses lock.
    """
    outputs = adapter.outputs
    valid = outputs.valid
    plot_time = outputs.uptime_epoch_ms[valid] * 1e-3
    doppler_freq_hz = outputs.doppler_freq_hz[valid]

    # Code phase accumulates without wrapping, so subtracting elapsed time leaves
    # only the departure from the nominal rate.
    residual_ms = outputs.code_phase_ms[valid] - outputs.uptime_epoch_ms[valid]
    if len(residual_ms):
        residual_ms = residual_ms - residual_ms[0]
    residual_chips = residual_ms * 1e-3 * adapter.signal.tracking_code_rate_chips_per_sec

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(plot_time, residual_chips, color="tab:purple", lw=2, label="Code delay")
    ax.set_ylabel("Code delay residual [chips]", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_xlabel("Uptime [s]")
    ax.grid(True)

    ax_doppler = ax.twinx()
    ax_doppler.plot(plot_time, doppler_freq_hz, color="tab:green", lw=2, label="Doppler")
    ax_doppler.set_ylabel("Doppler [Hz]", color="tab:green")
    ax_doppler.tick_params(axis="y", labelcolor="tab:green")

    ax.set_title(title if title is not None else f"Code delay and Doppler: {sig_id}")
    handles = [
        plt.Line2D([0], [0], color="tab:purple", lw=2, label="Code delay residual"),
        plt.Line2D([0], [0], color="tab:green", lw=2, label="Doppler"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    return ax


def plot_prompt_circ_length(
    fig: Figure | SubFigure,
    adapter: "TrackingChannelAdapter",
    sig_id: str,
    title: Optional[str] = None,
) -> Axes:
    """
    Prompt circular length per epoch, coloured by which carrier loop was running.

    Circular length is the coherence of the recent prompt history: the magnitude of
    the mean unit phasor, wrapped for Costas where the policy says so.  It is ~1
    when the prompt phase is steady and falls towards 0 as it scatters, which makes
    it the statistic the channel uses to decide the FLL has pulled the frequency
    error in far enough for the PLL to take over.

    Epochs filtered by the FLL and by the PLL are drawn in different colours, with
    the switching threshold and the handover instant both marked -- on a strong
    signal the FLL stretch is only a few epochs wide.  Expect a short FLL stretch while the loop
    pulls in, a crossing of the threshold, then PLL for the rest of the run.
    Dropping back towards the threshold under PLL is the signature of a channel
    about to lose lock; a channel that never leaves FLL never locked at all.
    """
    outputs = adapter.outputs
    valid = outputs.valid
    plot_time = outputs.uptime_epoch_ms[valid] * 1e-3
    circ_length = outputs.prompt_corr_circ_length[valid]
    pll = outputs.pll_mode[valid]

    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(plot_time[~pll], circ_length[~pll], s=4, color="tab:orange", label="FLL")
    ax.scatter(plot_time[pll], circ_length[pll], s=4, color="tab:blue", label="PLL")

    threshold = adapter.channel.loop_params.prompt_corr_circ_length_threshold
    ax.axhline(threshold, color="k", ls="--", lw=1.5,
               label=f"FLL -> PLL threshold ({threshold:g})")

    # On a strong signal the FLL stretch can be only a handful of epochs wide and
    # all but invisible as scattered points, so mark the handover explicitly.
    if pll.any() and not pll.all():
        handover_s = float(plot_time[np.argmax(pll)])
        ax.axvline(handover_s, color="k", ls=":", lw=1.5)
        ax.annotate(f"FLL -> PLL at {handover_s:.3f} s",
                    xy=(handover_s, 0.5), xytext=(6, 0), textcoords="offset points",
                    rotation=90, va="center", fontsize=8)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Prompt circular length")
    ax.set_xlabel("Uptime [s]")
    ax.set_title(title if title is not None else f"Carrier loop coherence: {sig_id}")
    ax.grid(True)
    ax.legend(markerscale=4, loc="lower right", fontsize=8)
    return ax
