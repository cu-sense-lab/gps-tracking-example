from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from . import bpsk_acquisition, tracking_channel
from .collect_metadata_utils import ExperimentMetadata


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
        freqs, psd_orig = scipy.signal.welch(
            orig_samples, fs=samp_rate, nperseg=nperseg, noverlap=noverlap,
            window="hann", return_onesided=False, scaling="density",
        )
        psd_orig = np.fft.fftshift(psd_orig)
        ax.plot(freqs / 1e6, 10 * np.log10(psd_orig), color="gray")

    ax.plot(freqs / 1e6, 10 * np.log10(psd), color="black")
    
    ax.set_title("Welch PSD Estimate of Raw Samples")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    ax.grid()
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
