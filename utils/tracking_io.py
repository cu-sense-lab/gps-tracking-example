"""
Saving and reloading tracking results.

Tracking a minute of 22 Msps samples takes minutes; forming a navigation solution
from the result takes seconds.  Notebook 01 therefore writes its channels here and
notebook 02 reads them back, so the expensive half runs once.

What is written is deliberately not a `TrackingChannel`.  A channel is a live
object with loop filters and a correlator mid-flight, and reconstructing one from
disk would invite the belief that tracking could be *resumed*, which it cannot --
the sample stream is gone.  What is written is the history: the output arrays, the
tap layout that gives them meaning, and enough provenance to refuse a file that
does not match what the caller thinks it is loading.

Arrays are sliced to `outputs.valid` on the way out.  `SignalTrackingOutputs`
pre-allocates to capacity and stops writing when full, so the tail of every array
is zeros; persisting those would put a spurious run of (0, 0) epochs into every
plot and, worse, into every pseudorange.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from . import tracking_channel

FORMAT_VERSION = 2

# Every per-epoch array in SignalTrackingOutputs, and every per-C/N0-hop array.
# Listed explicitly rather than discovered by introspection: a new output field
# should be a deliberate addition here, not something that silently starts (or
# stops) being persisted.
_EPOCH_FIELDS = (
    "uptime_epoch_ms",
    "carr_phase_errors_cycles",
    "code_phase_errors_chips",
    "subcarrier_offset_chips",
    "carr_phase_cycles",
    "doppler_freq_hz",
    "code_phase_ms",
    "delta_omega",
    "prompt_corr_circ_length",
    "pll_mode",
    "epoch_duration_ms",
    "overlay_synced",
    "bit_synced",
)
_CN0_FIELDS = ("cn0_dbhz", "cn0_uptime_ms")


@dataclass
class TrackingResult:
    """One channel's tracking history, read back from disk."""

    signal_id: str
    signal_type_id: str
    outputs: tracking_channel.SignalTrackingOutputs
    """A real `SignalTrackingOutputs`, sized exactly to the epochs written, so
    everything that consumes tracking outputs -- plotting, `utils.nav.symbols` --
    works on a loaded result without knowing it came from a file."""

    acquisition_doppler_hz: float
    acquisition_code_phase_ms: float

    @property
    def duration_ms(self) -> float:
        uptime = self.outputs.uptime_epoch_ms
        return float(uptime[-1] - uptime[0]) if len(uptime) > 1 else 0.0


@dataclass
class TrackingRun:
    """Everything one call to `save_tracking_results` recorded."""

    results: dict[str, TrackingResult]
    collect_id: str
    signal_type_id: str
    samp_rate: float
    experiment_name: str
    git_commit: str
    format_version: int

    def __getitem__(self, signal_id: str) -> TrackingResult:
        return self.results[signal_id]

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def signal_ids(self) -> list[str]:
        return sorted(self.results)

    @property
    def shortest_duration_ms(self) -> float:
        return min((r.duration_ms for r in self.results.values()), default=0.0)


def _git_commit() -> str:
    """Current commit, or a marker when this is not a checkout.

    Provenance, not a dependency: a tracking file that outlives the code that made
    it is much easier to explain when it says which commit made it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_tracking_results(
    path: str | Path,
    channels: dict,
    *,
    collect_id: str,
    signal_type_id: str,
    samp_rate: float,
    experiment_name: str = "",
    acquisition_results: dict | None = None,
) -> Path:
    """
    Write every channel's tracking history to one HDF5 file.

    `channels` maps signal id to either a `TrackingChannelAdapter` or a bare
    `TrackingChannel` -- both are accepted because notebook 01 holds adapters while
    the tests hold channels, and requiring one to wrap or unwrap would be noise.

    Returns the path written, and creates parent directories.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["collect_id"] = collect_id
        f.attrs["signal_type_id"] = signal_type_id
        f.attrs["samp_rate"] = float(samp_rate)
        f.attrs["experiment_name"] = experiment_name
        f.attrs["git_commit"] = _git_commit()

        for signal_id, holder in channels.items():
            channel = getattr(holder, "channel", holder)
            outputs = channel.outputs
            valid = outputs.valid
            group = f.create_group(signal_id)

            for name in _EPOCH_FIELDS:
                group.create_dataset(
                    name, data=np.asarray(getattr(outputs, name))[valid], compression="gzip"
                )
            # Correlators are (epochs, taps, components) complex.
            group.create_dataset("corr", data=outputs.corr[valid], compression="gzip")

            cn0_valid = outputs.cn0_valid
            for name in _CN0_FIELDS:
                group.create_dataset(
                    name, data=np.asarray(getattr(outputs, name))[cn0_valid], compression="gzip"
                )

            layout = outputs.tap_layout
            group.attrs["tap_code_offsets"] = np.asarray(layout.code_offsets_chips)
            group.attrs["tap_subcarrier_offsets"] = np.asarray(layout.subcarrier_offsets_chips)
            # Roles are (name, index) pairs; HDF5 attributes take flat arrays, so
            # the two halves are stored side by side and zipped back on load.
            group.attrs["tap_role_names"] = [name for name, _ in layout.roles]
            group.attrs["tap_role_indices"] = np.asarray([i for _, i in layout.roles])
            group.attrs["num_components"] = outputs.num_components

            # `bpsk_acquisition.AcquisitionResult` exposes the seed through
            # `acq_doppler_hz` / `acq_code_phase_ms`, which already fold in the fine
            # search when one ran.  Recorded because notebook 02 wants to compare
            # the decoded time of week against the code phase acquisition found --
            # for L5 that phase is absolute modulo NH20's 20 ms, which is an
            # independent check on the decode.
            acq = (acquisition_results or {}).get(signal_id)
            group.attrs["acquisition_doppler_hz"] = float(
                getattr(acq, "acq_doppler_hz", np.nan) if acq is not None else np.nan
            )
            group.attrs["acquisition_code_phase_ms"] = float(
                getattr(acq, "acq_code_phase_ms", np.nan) if acq is not None else np.nan
            )
    return path


def load_tracking_results(path: str | Path) -> TrackingRun:
    """
    Read a file written by `save_tracking_results`.

    Raises on a format version this code does not understand rather than reading
    what it can.  A partially-understood tracking file produces a plausible
    navigation solution that is wrong, which is the worst possible failure here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no tracking results at {path}. Run notebook 01 to completion first -- "
            "its final cell writes this file."
        )

    with h5py.File(path, "r") as f:
        version = int(f.attrs.get("format_version", -1))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} is tracking-file format version {version}, but this code "
                f"reads version {FORMAT_VERSION}. Re-run notebook 01 to rewrite it."
            )

        results: dict[str, TrackingResult] = {}
        signal_type_id = str(f.attrs["signal_type_id"])
        for signal_id in f:
            group = f[signal_id]
            num_epochs = len(group["uptime_epoch_ms"])
            num_cn0 = len(group["cn0_uptime_ms"])
            num_components = int(group.attrs["num_components"])

            layout = tracking_channel.TapLayout(
                code_offsets_chips=tuple(group.attrs["tap_code_offsets"]),
                subcarrier_offsets_chips=tuple(group.attrs["tap_subcarrier_offsets"]),
                roles=tuple(
                    (str(name), int(index))
                    for name, index in zip(
                        group.attrs["tap_role_names"], group.attrs["tap_role_indices"]
                    )
                ),
            )
            outputs = tracking_channel.SignalTrackingOutputs(
                capacity=num_epochs,
                num_components=num_components,
                cn0_capacity=num_cn0,
                tap_layout=layout,
            )
            for name in _EPOCH_FIELDS:
                getattr(outputs, name)[:] = group[name][:]
            outputs.corr[:] = group["corr"][:]
            for name in _CN0_FIELDS:
                getattr(outputs, name)[:] = group[name][:]
            # The arrays were allocated at exactly the written length, so the whole
            # of each is valid -- which is what these two indices assert.
            outputs.output_index = num_epochs
            outputs.cn0_index = num_cn0

            results[signal_id] = TrackingResult(
                signal_id=signal_id,
                signal_type_id=signal_type_id,
                outputs=outputs,
                acquisition_doppler_hz=float(group.attrs["acquisition_doppler_hz"]),
                acquisition_code_phase_ms=float(group.attrs["acquisition_code_phase_ms"]),
            )

        return TrackingRun(
            results=results,
            collect_id=str(f.attrs["collect_id"]),
            signal_type_id=signal_type_id,
            samp_rate=float(f.attrs["samp_rate"]),
            experiment_name=str(f.attrs.get("experiment_name", "")),
            git_commit=str(f.attrs.get("git_commit", "unknown")),
            format_version=version,
        )


def require_tracking_run(
    path: str | Path,
    *,
    collect_id: str | None = None,
    signal_type_id: str | None = None,
    minimum_duration_ms: float = 0.0,
    minimum_signals: int = 0,
) -> TrackingRun:
    """
    Load, and fail with an actionable message if the file is not what is needed.

    Written for notebook 02, where the alternative to checking is a traceback deep
    inside a pseudorange calculation an hour later.  Each check names the setting in
    notebook 01 that fixes it.
    """
    run = load_tracking_results(path)

    if collect_id is not None and run.collect_id != collect_id:
        raise ValueError(
            f"{path} holds tracking for collect {run.collect_id!r}, but this notebook "
            f"is configured for {collect_id!r}. Re-run notebook 01 with the same "
            "EXPERIMENT_INDEX, or point this notebook at the other collect."
        )
    if signal_type_id is not None and run.signal_type_id != signal_type_id:
        raise ValueError(
            f"{path} holds {run.signal_type_id} tracking, but this notebook is "
            f"configured for {signal_type_id}. Re-run notebook 01 with "
            f"SIGNAL_ID set to {signal_type_id}."
        )
    if len(run) < minimum_signals:
        raise ValueError(
            f"{path} holds only {len(run)} tracked signal(s) but {minimum_signals} "
            "are needed for a position fix. Acquisition found too few satellites -- "
            "try a longer acquisition dwell, or a different point in the collect."
        )
    if run.shortest_duration_ms < minimum_duration_ms:
        raise ValueError(
            f"{path} holds only {run.shortest_duration_ms / 1000:.1f} s of tracking on "
            f"its shortest channel, and {minimum_duration_ms / 1000:.1f} s are needed "
            "to decode the navigation message. Re-run notebook 01 with a larger "
            "TRACK_DURATION_MS."
        )
    return run
