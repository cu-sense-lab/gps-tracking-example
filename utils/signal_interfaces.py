"""
GPS signal catalog: Link, Signal, and the acquisition/tracking policies that
configure how each signal is acquired and tracked.

Terminology (see TODO_SIGNALS.md): a **link** is the RF carrier (L1, L2, L5);
a **signal** is a particular modulation on a link, for one constellation (GPS
L1 C/A, GPS L2C, GPS L5). A signal's carrier frequency can differ from its
link's nominal frequency for a BOC sidelobe -- not the case for any signal
built here yet, but `Signal.carrier_freq_hz` is independent of `Link` for when
it is.

A `Signal` subclass -- `GpsL1CA`, `GpsL2C`, `GpsL5` -- IS the class-based
definition of one signal: instantiating it for a PRN builds that satellite's
spreading-code components (see `utils.code_components`) into a `CodeSet`.
Acquisition and tracking are configured separately, in `ACQUISITION_POLICIES`
and `TRACKING_POLICIES` below, keyed by each signal's `signal_type_id` --
acquisition typically locks onto a single component (L2C acquires on CM alone
at 511.5 kcps, even though tracking later uses CM and CL together), and
discriminator/loop policy is a tracking-strategy choice independent of what
the signal itself is built of. Neither belongs on the signal definition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Iterable

import numpy as np

import gnss_tools.signals.gps_l1ca as gps_l1ca
import gnss_tools.signals.gps_l2c as gps_l2c
import gnss_tools.signals.gps_l5 as gps_l5

from . import bpsk_acquisition
from . import sample_streaming
from . import tracking_channel
from .code_components import (
    BPSKComponent,
    CodeComponent,
    CodeSet,
    QPSKComponent,
    TDBPSKComponent,
    build_code_set,
)


class LinkId(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L5 = "L5"


@dataclass(frozen=True)
class Link:
    """One RF carrier, shared by every signal modulated onto it."""

    id: LinkId
    nominal_freq_hz: float


LINKS: dict[LinkId, Link] = {
    LinkId.L1: Link(LinkId.L1, gps_l1ca.CARRIER_FREQ),
    LinkId.L2: Link(LinkId.L2, gps_l2c.CARRIER_FREQ),
    LinkId.L5: Link(LinkId.L5, gps_l5.CARRIER_FREQ),
}


class SignalType(StrEnum):
    """
    The overall modulation a signal is built from (see TODO_SIGNALS.md).

    Most of these correlate with the plain BPSK kernel -- a QPSK signal's I/Q
    components each use it independently, and (per the TODO) a TDBPSK signal's
    time-multiplexing is already handled generically by `code_components`.
    Only BOC/TMBOC/CBOC's subcarrier structure needs a specialized correlator,
    which does not exist yet (see `BOCComponent`).
    """

    BPSK = "BPSK"
    TDBPSK = "TDBPSK"
    QPSK = "QPSK"
    BOC = "BOC"
    TMBOC = "TMBOC"
    CBOC = "CBOC"


class Signal(ABC):
    """
    A particular signal definition on a link, for one satellite.

    Carries only what the signal *is*: which link it rides on, its carrier
    frequency, and its spreading code components (built fresh per PRN, since
    only the code sequences vary by satellite). Acquisition and tracking
    policy live outside this class entirely -- see `ACQUISITION_POLICIES`/
    `TRACKING_POLICIES` below.
    """

    signal_id: ClassVar[str]
    """Unique id for this signal type (constellation + signal), e.g. "GPS_L1CA".
    Matches the convention in submodules/gnss-tools/gnss_tools/signals/catalog.py.
    This is what the policy dictionaries below key on -- NOT `signal_id`, which
    is the per-satellite id ("G01") assigned below."""

    link: ClassVar[Link]
    signal_type: ClassVar[SignalType]
    carrier_freq_hz: ClassVar[float]
    tracking_code_rate_chips_per_sec: ClassVar[float]
    # Duration of one full pass through the primary code. Tiered-code
    # integration accumulates in units of it.
    primary_period_ms: ClassVar[int] = 1

    def __init__(self, prn: int) -> None:
        self.prn = prn
        self.signal_id = f"G{prn:02d}"
        self.code_set: CodeSet = build_code_set(self._build_components(prn))

    @staticmethod
    @abstractmethod
    def _build_components(prn: int) -> list[CodeComponent]:
        """This signal's spreading code components for one PRN."""

    @property
    def component_names(self) -> tuple[str, ...]:
        return self.code_set.names


class GpsL1CA(Signal):
    """GPS L1 C/A: one BPSK component, 1.023 Mcps Gold code."""

    signal_id = "GPS_L1CA"
    link = LINKS[LinkId.L1]
    signal_type = SignalType.BPSK
    carrier_freq_hz = gps_l1ca.CARRIER_FREQ
    tracking_code_rate_chips_per_sec = gps_l1ca.CODE_RATE
    primary_period_ms = 1

    @staticmethod
    def _build_components(prn: int) -> list[CodeComponent]:
        code = (1 - 2 * gps_l1ca.get_GPS_L1CA_code_sequence(prn)).astype(np.int8)
        return [BPSKComponent(name="CA", sequence=code)]


class GpsL2C(Signal):
    """
    GPS L2C: CM and CL, time-division multiplexed on the 1.023 Mcps combined
    clock (each contributes one component chip every other chip).
    """

    signal_id = "GPS_L2C"
    link = LINKS[LinkId.L2]
    signal_type = SignalType.TDBPSK
    carrier_freq_hz = gps_l2c.CARRIER_FREQ
    tracking_code_rate_chips_per_sec = gps_l2c.CODE_RATE_L2CLM
    # CM repeats every 20 ms; CL every 1.5 s.
    primary_period_ms = 20

    @staticmethod
    def _build_components(prn: int) -> list[CodeComponent]:
        code_cm = (1 - 2 * gps_l2c.get_GPS_L2CM_code_sequence(prn)).astype(np.int8)
        code_cl = (1 - 2 * gps_l2c.get_GPS_L2CL_code_sequence(prn)).astype(np.int8)
        return [
            TDBPSKComponent(name="CM", sequence=code_cm, chips_per_component_chip=2, component_offset_chips=0),
            TDBPSKComponent(name="CL", sequence=code_cl, chips_per_component_chip=2, component_offset_chips=1),
        ]


class GpsL5(Signal):
    """
    GPS L5: I and Q, co-located on every chip and separated by carrier
    phase (a QPSK pair), at 10.23 Mcps. Each carries a Neuman-Hoffman overlay
    -- NH10 on I, NH20 on Q -- that advances one chip per 1 ms primary code
    period; NH10's period is exactly the 10 ms CNAV symbol, so locking NH20
    (the pilot) also gives symbol sync on I for free (see
    `utils.secondary_code`).
    """

    signal_id = "GPS_L5"
    link = LINKS[LinkId.L5]
    signal_type = SignalType.QPSK
    carrier_freq_hz = gps_l5.CARRIER_FREQ
    tracking_code_rate_chips_per_sec = gps_l5.CODE_RATE
    # 10230 chips at 10.23 Mcps -- the aligned correlator's existing interval
    # granularity, which is what makes tiered-code wipe-off cheap.
    primary_period_ms = 1

    @staticmethod
    def _build_components(prn: int) -> list[CodeComponent]:
        # NOTE: unlike the L1CA/L2C getters, these return float64 0/1 rather than int8.
        code_i = (1 - 2 * gps_l5.get_GPS_L5I_code_sequence(prn)).astype(np.int8)
        code_q = (1 - 2 * gps_l5.get_GPS_L5Q_code_sequence(prn)).astype(np.int8)
        overlay_i = (1 - 2 * gps_l5.NEUMAN_HOFFMAN_SEQ_L5I).astype(np.int8)
        overlay_q = (1 - 2 * gps_l5.NEUMAN_HOFFMAN_SEQ_L5Q).astype(np.int8)
        return [
            QPSKComponent(name="I", sequence=code_i, overlay=overlay_i),
            QPSKComponent(name="Q", sequence=code_q, overlay=overlay_q),
        ]


SIGNALS: dict[str, type[Signal]] = {
    GpsL1CA.signal_id: GpsL1CA,
    GpsL2C.signal_id: GpsL2C,
    GpsL5.signal_id: GpsL5,
}


# --------------------------------------------------------------------------
# Acquisition and tracking policy -- kept separate from the signal
# definitions above (see TODO_SIGNALS.md and the module docstring).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AcquisitionPolicy:
    """Which single component a signal is acquired on."""

    component_name: str


ACQUISITION_POLICIES: dict[str, AcquisitionPolicy] = {
    "GPS_L1CA": AcquisitionPolicy(component_name="CA"),
    "GPS_L2C": AcquisitionPolicy(component_name="CM"),
    # Acquire on Q (pilot). At 1 ms coherent integration I and Q perform
    # identically -- Q's Neuman-Hoffman overlay flips sign every 1 ms just as
    # I's CNAV symbols do -- but acquiring on the component that will
    # ultimately drive the loops keeps the code phase reference consistent.
    "GPS_L5": AcquisitionPolicy(component_name="Q"),
}


@dataclass(frozen=True)
class TrackingPolicy:
    """
    Which components drive the tracking loops, and how tiered-code sync
    changes that.

    `synced_discriminator_policy`/`synced_coherent_periods` apply once a
    tiered (overlay) code is synchronised and can be wiped off:
    `synced_coherent_periods` is how many primary code periods are then folded
    into one coherent accumulation. Defaults leave behaviour unchanged for
    signals without an overlay.
    """

    discriminator_policy: tracking_channel.LoopDiscriminatorPolicy
    synced_discriminator_policy: tracking_channel.LoopDiscriminatorPolicy | None = None
    synced_coherent_periods: int = 1


TRACKING_POLICIES: dict[str, TrackingPolicy] = {
    "GPS_L1CA": TrackingPolicy(
        discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=0, code_components=(0,), costas=True
        ),
    ),
    # Only CM drives the loops. Adding CL to code_components would combine
    # both components non-coherently in the DLL -- a genuine improvement, but
    # a behavioural change that belongs in its own commit rather than riding
    # along with this refactor.
    "GPS_L2C": TrackingPolicy(
        discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=0, code_components=(0,), costas=True
        ),
    ),
    "GPS_L5": TrackingPolicy(
        # Carrier runs on Q, the pilot, from the start. I and Q are in
        # quadrature, so switching the carrier component mid-track (e.g. at
        # overlay sync) would force the PLL to re-pull; starting on Q avoids
        # that. Costas stays True until the overlay is stripped: NH20 flips
        # Q's sign every 1 ms, so the pilot is not yet effectively dataless.
        #
        # The delay discriminator combines I and Q non-coherently. They are
        # equal power, so this is worth ~3 dB and costs nothing: overlay and
        # data flips cancel in the magnitudes.
        discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=1, code_components=(0, 1), costas=True
        ),
        # Once NH is stripped, Q is genuinely dataless: the phase
        # discriminator can use the full four-quadrant angle instead of
        # wrapping at +/-1/4 cycle, which is where the pilot's ~6 dB comes
        # from.
        #
        # The delay discriminator drops to Q alone. I is capped at 10 ms by
        # its CNAV symbols, so over a 20 ms epoch it would partly cancel and
        # drag the combined magnitude down rather than help.
        synced_discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=1, code_components=(1,), costas=False
        ),
        # NH20's period. The loop update rate drops to 20 ms with it, so the
        # loop filter is retuned at the same moment (see TrackingChannel).
        synced_coherent_periods=20,
    ),
}


# --------------------------------------------------------------------------
# Runtime: tracking channel adapter, and the module-level helpers that tie a
# signal type's components together with its (separately defined) policies.
# --------------------------------------------------------------------------


@dataclass
class TrackingChannelAdapter:
    """Pairs a tracking channel with the signal that configured it."""

    signal: Signal
    channel: tracking_channel.TrackingChannel

    @property
    def outputs(self) -> tracking_channel.SignalTrackingOutputs:
        return self.channel.outputs

    def process_sample_buffer(self, sample_buffer: sample_streaming.SampleBuffer) -> None:
        self.channel.process_sample_buffer(sample_buffer)

    def set_mode_pll(self) -> None:
        self.channel.loop_state.mode = tracking_channel.TrackingLoopMode.PLL

    def component_index(self, name: str) -> int:
        """Index of a named component, e.g. "CM"/"CL" for L2C."""
        return self.signal.code_set.index_of(name)

    def get_prompt_component(self, component: int = 0) -> np.ndarray:
        return self.outputs.prompt_corr[self.outputs.valid, component]

    def get_early_component(self, component: int = 0) -> np.ndarray:
        return self.outputs.early_corr[self.outputs.valid, component]

    def get_late_component(self, component: int = 0) -> np.ndarray:
        return self.outputs.late_corr[self.outputs.valid, component]


def build_signals(signal_type: type[Signal], prns: Iterable[int] = range(1, 33)) -> dict[str, Signal]:
    return {f"G{prn:02d}": signal_type(prn) for prn in prns}


def build_acquisition_code_params(
    signal_type: type[Signal],
    signals: dict[str, Signal],
) -> dict[str, bpsk_acquisition.AcqSignalCodeParameters]:
    policy = ACQUISITION_POLICIES[signal_type.signal_id]
    params: dict[str, bpsk_acquisition.AcqSignalCodeParameters] = {}
    for signal_id, signal in signals.items():
        component = signal.code_set.components[signal.code_set.index_of(policy.component_name)]
        params[signal_id] = bpsk_acquisition.AcqSignalCodeParameters(
            rate_chips_per_sec=signal_type.tracking_code_rate_chips_per_sec / component.chips_per_component_chip,
            length_chips=component.code_length,
            sequence=component.sequence,
            # True when the acquisition component is one of several
            # sharing the chip clock by time-division (e.g. L2C's CM/CL):
            # the replica must be zero-filled on the other component's
            # chips rather than holding this component's value across
            # both, so it doesn't spuriously correlate against the other
            # code.
            is_interleaved=isinstance(component, TDBPSKComponent),
        )
    return params


def create_tracking_channels(
    signal_type: type[Signal],
    signals: dict[str, Signal],
    acquisition_results: dict[str, bpsk_acquisition.AcquisitionResult],
    tracking_signal_ids: Iterable[str],
    loop_params: tracking_channel.TrackingLoopParameters,
    output_capacity: int,
    start_mode_pll: bool = False,
) -> dict[str, TrackingChannelAdapter]:
    policy = TRACKING_POLICIES[signal_type.signal_id]
    channels: dict[str, TrackingChannelAdapter] = {}
    for signal_id in tracking_signal_ids:
        signal = signals[signal_id]
        acq_result = acquisition_results[signal_id]

        code_rate_ms_per_sec = (
            1.0 + acq_result.acq_doppler_hz / signal_type.carrier_freq_hz
        ) * 1e3
        initial_state = tracking_channel.TrackingSignalState(
            uptime_epoch_ms=acq_result.uptime_epoch_ms,
            code_phase_ms=acq_result.acq_code_phase_seconds * 1e3,
            code_rate_ms_per_sec=code_rate_ms_per_sec,
            carrier_phase_cycles=0.0,
            carrier_rate_cyc_per_sec=acq_result.acq_doppler_hz,
        )
        signal_params = tracking_channel.TrackingSignalParameters(
            code_set=signal.code_set,
            nominal_code_rate_chips_per_sec=signal_type.tracking_code_rate_chips_per_sec,
            carrier_freq_hz=signal_type.carrier_freq_hz,
            primary_period_ms=signal_type.primary_period_ms,
        )
        channel = tracking_channel.TrackingChannel(
            loop_params=loop_params,
            signal_params=signal_params,
            initial_signal_state=initial_state,
            output_capacity=output_capacity,
            discriminator_policy=policy.discriminator_policy,
            synced_policy=policy.synced_discriminator_policy,
            synced_coherent_periods=policy.synced_coherent_periods,
        )

        adapter = TrackingChannelAdapter(signal=signal, channel=channel)
        if start_mode_pll:
            adapter.set_mode_pll()
        channels[signal_id] = adapter
    return channels
