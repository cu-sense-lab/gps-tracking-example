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

What acquisition *cannot* locate is policy too.  A signal whose components
have unequal periods is only pinned modulo the shorter one, and the leftover
is resolved after the dwell rather than during it -- see `AcquisitionPolicy.
ambiguous_component`, `build_ambiguity_search` and
`resolve_acquisition_ambiguities`.  The plan itself is built per PRN because
it correlates that satellite's long code, but which component is ambiguous is
a property of how the signal is acquired, so it lives with the policy.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Iterable

import numpy as np

import gnss_tools.signals.gps_l1ca as gps_l1ca
import gnss_tools.signals.gps_l2c as gps_l2c
import gnss_tools.signals.gps_l5 as gps_l5

from . import ambiguity_resolution
from . import bpsk_acquisition
from . import sample_streaming
from . import tracking_channel
from .code_components import (
    Branch,
    CodeComponent,
    CodeSet,
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


def _symbol_period_ms(signal_module) -> int:
    """Data symbol duration in ms: 20 for L1 C/A and L2 CNAV, 10 for L5 CNAV."""
    period_ms = 1e3 / signal_module.DATA_SYMBOL_RATE
    if period_ms != int(period_ms):
        raise ValueError(f"symbol period {period_ms} ms is not a whole millisecond")
    return int(period_ms)


def _interleave(sequences: list[np.ndarray]) -> list[np.ndarray]:
    """
    Place each sequence on its own chip of a shared clock, zero elsewhere.

    L2C's CM and CL alternate chip by chip on the 1.023 Mcps combined clock
    (IS-GPS-200), so CM comes back as (CM_0, 0, CM_1, 0, ...) and CL as
    (0, CL_0, 0, CL_1, ...).  Stating every component on the signal's own chip
    axis is what lets one correlator serve every signal -- see
    `utils.code_components`.
    """
    stride = len(sequences)
    filled = []
    for offset, sequence in enumerate(sequences):
        padded = np.zeros(len(sequence) * stride, dtype=np.int8)
        padded[offset::stride] = sequence
        filled.append(padded)
    return filled


def _tiered_code(primary: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Fold a tiered code into one long sequence: the primary repeated once per
    overlay chip, each repetition signed by that chip.

    Putting the overlay *in the replica* is what lets acquisition integrate across
    it instead of being broken by it -- and the resulting code phase then carries
    the overlay phase for free, since the position within the long code says which
    overlay chip is in play.
    """
    return np.concatenate([primary * int(sign) for sign in overlay]).astype(np.int8)


class Signal(ABC):
    """
    A particular signal definition on a link, for one satellite.

    Carries only what the signal *is*: which link it rides on, its carrier
    frequency, and its spreading code components (built fresh per PRN, since
    only the code sequences vary by satellite). Acquisition and tracking
    policy live outside this class entirely -- see `ACQUISITION_POLICIES`/
    `TRACKING_POLICIES` below.
    """

    signal_type_id: ClassVar[str]
    """Unique id for this signal type (constellation + signal), e.g. "GPS_L1CA".
    Matches the convention in submodules/gnss-tools/gnss_tools/signals/catalog.py.
    This is what the policy dictionaries below key on -- NOT `signal_id`, which
    is the per-satellite id ("G01") assigned below."""

    link: ClassVar[Link]
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

    @property
    def overlay_period_ms(self) -> int:
        """Modulus of the shared tiered-code counter; 0 when there is no overlay."""
        overlays = [c.overlay for c in self.code_set.components if c.overlay is not None]
        if not overlays:
            return 0
        return int(np.lcm.reduce([len(o) for o in overlays])) * self.primary_period_ms


class GpsL1CA(Signal):
    """GPS L1 C/A: one BPSK component, 1.023 Mcps Gold code."""

    signal_type_id = "GPS_L1CA"
    link = LINKS[LinkId.L1]
    carrier_freq_hz = gps_l1ca.CARRIER_FREQ
    tracking_code_rate_chips_per_sec = gps_l1ca.CODE_RATE
    primary_period_ms = 1

    @staticmethod
    def _build_components(prn: int) -> list[CodeComponent]:
        code = (1 - 2 * gps_l1ca.get_GPS_L1CA_code_sequence(prn)).astype(np.int8)
        return [
            CodeComponent(
                name="CA", sequence=code, branch=Branch.Q,
                symbol_period_ms=_symbol_period_ms(gps_l1ca),
            )
        ]


class GpsL2C(Signal):
    """
    GPS L2C: CM and CL, time-division multiplexed chip by chip on the
    1.023 Mcps combined clock -- CM on even chips, CL on odd, each zero-filled
    on the other's.  Both ride the quadrature branch.
    """

    signal_type_id = "GPS_L2C"
    link = LINKS[LinkId.L2]
    carrier_freq_hz = gps_l2c.CARRIER_FREQ
    tracking_code_rate_chips_per_sec = gps_l2c.CODE_RATE_L2CLM
    # CM repeats every 20 ms; CL every 1.5 s.
    primary_period_ms = 20

    @staticmethod
    def _build_components(prn: int) -> list[CodeComponent]:
        code_cm = (1 - 2 * gps_l2c.get_GPS_L2CM_code_sequence(prn)).astype(np.int8)
        code_cl = (1 - 2 * gps_l2c.get_GPS_L2CL_code_sequence(prn)).astype(np.int8)
        cm, cl = _interleave([code_cm, code_cl])
        return [
            CodeComponent(
                name="CM", sequence=cm, branch=Branch.Q,
                symbol_period_ms=_symbol_period_ms(gps_l2c),
            ),
            # CL is the dataless pilot: no symbol period, so nothing caps its
            # coherent integration but Doppler.  Same branch as CM -- both sit in
            # quadrature to P(Y) -- which is what makes the carrier handover at
            # ambiguity resolution phase-continuous.
            CodeComponent(name="CL", sequence=cl, branch=Branch.Q),
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

    signal_type_id = "GPS_L5"
    link = LINKS[LinkId.L5]
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
            CodeComponent(
                name="I", sequence=code_i, branch=Branch.I, overlay=overlay_i,
                symbol_period_ms=_symbol_period_ms(gps_l5),
            ),
            # Q is the dataless pilot: no symbol period.  Opposite branch to I,
            # so moving the carrier loop between them would cost a quarter-cycle
            # re-pull -- which is why it starts on Q and stays there.
            CodeComponent(name="Q", sequence=code_q, branch=Branch.Q, overlay=overlay_q),
        ]


SIGNALS: dict[str, type[Signal]] = {
    GpsL1CA.signal_type_id: GpsL1CA,
    GpsL2C.signal_type_id: GpsL2C,
    GpsL5.signal_type_id: GpsL5,
}


# --------------------------------------------------------------------------
# Acquisition and tracking policy -- kept separate from the signal
# definitions above (see TODO_SIGNALS.md and the module docstring).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AcquisitionPolicy:
    """Which single component a signal is acquired on, and how."""

    component_name: str

    # Fold the component's tiered (overlay) code into the replica rather than
    # letting it break coherence.  That lengthens the replica by the overlay's
    # period, and the recovered code phase then carries the overlay phase for
    # free -- the position within the long code says which overlay chip is in
    # play.  See `acquisition_resolves_overlay_phase`.
    include_overlay: bool = False

    # A component whose period is longer than the acquisition code's, and which
    # acquisition therefore leaves unlocated: the dwell pins the signal only
    # modulo the code it correlated against.  `build_ambiguity_search` turns
    # this into a hypothesis search over the same dwell; None means acquisition
    # locates the whole signal (L1 C/A).
    ambiguous_component: str | None = None


ACQUISITION_POLICIES: dict[str, AcquisitionPolicy] = {
    "GPS_L1CA": AcquisitionPolicy(component_name="CA"),
    # Acquisition locates CM's 20 ms period, which leaves CL -- 75 times longer
    # -- unlocated.  Until the search below runs, the channel generates CL from
    # whatever phase CM supplied, i.e. as though CL were at its code origin,
    # which is right 1 time in 75.
    "GPS_L2C": AcquisitionPolicy(component_name="CM", ambiguous_component="CL"),
    # Acquire on Q x NH20 -- the pilot, with its overlay folded into the
    # replica.  I x NH10 is half the length and was the obvious choice, but it
    # has a degeneracy: L5I's CNAV symbol period EQUALS NH10's, so the data
    # reduces to a sign on each whole composite period and a dwell whose blocks
    # straddle a symbol boundary can score a neighbouring NH10 index above the
    # true one (measured: 8.31 ms recovered for a true 7.31 ms).  Q carries no
    # data at all, so its composite is exact over the full 20 ms.
    #
    # It costs 4x the search grid -- twice the code phase cells and, because the
    # 20 ms replica halves the Doppler bin, twice the Doppler bins.  It buys:
    #
    #   * the shared overlay counter outright.  A 20 ms code phase pins the
    #     counter mod 20, and NH10's index is that mod 10, so BOTH overlays are
    #     resolved and no hypothesis search is needed at all.
    #   * ~1.5 dB, because no data flip breaks coherence inside a block
    #     (measured on real L5: G14 +16.9 -> +18.7, G01 +12.8 -> +14.5,
    #     G08 +1.9 -> +3.4 dB of margin over threshold).
    "GPS_L5": AcquisitionPolicy(component_name="Q", include_overlay=True),
}


@dataclass(frozen=True)
class TrackingPolicy:
    """
    Which components drive the tracking loops, and how tiered-code sync and
    ambiguity resolution change that.

    `synced_discriminator_policy`/`synced_coherent_integration_ms` apply once a
    tiered (overlay) code is synchronised and can be wiped off:
    `synced_coherent_integration_ms` is how long one coherent accumulation then
    lasts. Defaults leave behaviour unchanged for signals without an overlay.

    `resolved_discriminator_policy` applies instead of `discriminator_policy`
    when the long-code ambiguity was resolved, i.e. when a component
    acquisition could not locate is now usable.  Distinct from `synced_*`,
    which is about stripping a tiered code: L2C's CL is not correlatable at all
    until its block is known, whereas L5's Q is always correlatable and only
    becomes *dataless* once NH is wiped off.
    """

    discriminator_policy: tracking_channel.LoopDiscriminatorPolicy
    synced_discriminator_policy: tracking_channel.LoopDiscriminatorPolicy | None = None
    synced_coherent_integration_ms: int = tracking_channel.CORRELATION_INTERVAL_MS
    resolved_discriminator_policy: tracking_channel.LoopDiscriminatorPolicy | None = None


TRACKING_POLICIES: dict[str, TrackingPolicy] = {
    "GPS_L1CA": TrackingPolicy(
        discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=0, code_components=(0,), costas=True
        ),
    ),
    "GPS_L2C": TrackingPolicy(
        # Fallback, used when the CL block is unknown: CM alone, Costas on.  CL
        # cannot be generated at the right phase, so it would feed noise into
        # both loops.
        discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=0, code_components=(0,), costas=True
        ),
        # Once the block is resolved, CL is a genuine dataless pilot: the
        # carrier loop moves to it and drops Costas wrapping for the full
        # four-quadrant angle, and the delay discriminator combines both
        # equal-power components non-coherently.  Unlike L5's I/Q, CM and CL are
        # time-multiplexed on one carrier branch (both quadrature to P(Y)), so
        # moving the carrier reference between them is phase-continuous and costs
        # no re-pull -- see `CodeSet.share_branch`.
        resolved_discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=1, code_components=(0, 1), costas=False
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
        # The delay discriminator keeps both components.  Capping the epoch at
        # I's 10 ms CNAV symbol is what makes that possible: I now comes out at
        # full magnitude rather than half-cancelled across two symbols, so
        # combining two equal-power components non-coherently is worth ~3 dB.
        # Data flips do not matter here -- the DLL combines magnitudes.
        #
        # Only the *carrier* loop moves to the pilot alone, which is where the
        # dataless four-quadrant discriminator pays.
        synced_discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=1, code_components=(0, 1), costas=False
        ),
        # One CNAV symbol, and the whole signal's limit.
        #
        # NH20's 20 ms period is the tempting choice and buys another 3 dB on the
        # pilot, but one epoch serves every component: Q is dataless and would take
        # it, while I would span two CNAV symbols and cancel.  10 ms is exactly one
        # symbol on I, exactly half an NH20 period on Q -- and since epochs anchor
        # to multiples of their own length in code phase, they keep tiling NH20
        # evenly.  It also leaves I symbol-synchronous, which is what a CNAV
        # decoder wants.
        #
        # The loop update rate drops to 10 ms with it, so the loop filter is
        # retuned at the same moment (see TrackingChannel).
        synced_coherent_integration_ms=10,
    ),
}


# --------------------------------------------------------------------------
# What acquisition leaves ambiguous, and how it is resolved.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AmbiguitySearchPlan:
    """
    How to resolve what acquisition left ambiguous, for one signal.

    `code_set` is correlated as transmitted and only `scored_component` is read.
    `hypothesis_stride_chips` is the acquisition code's period on the signal's chip
    clock, so hypothesis h tests "the dwell is h acquisition periods into the long
    code".
    """

    code_set: CodeSet
    scored_component: int
    num_hypotheses: int
    hypothesis_stride_chips: float
    code_rate_chips_per_sec: float
    description: str


def acquisition_code(signal_type: type[Signal], signal: Signal) -> np.ndarray:
    """The replica sequence acquisition correlates against, on its own chip clock."""
    policy = ACQUISITION_POLICIES[signal_type.signal_type_id]
    component = signal.code_set.components[
        signal.code_set.index_of(policy.component_name)
    ]
    if not policy.include_overlay:
        return component.sequence
    if component.overlay is None:
        raise ValueError(
            f"{signal_type.signal_type_id}: acquisition policy asks to fold in an "
            f"overlay, but component {policy.component_name!r} has none"
        )
    return _tiered_code(component.sequence, component.overlay)


def acquisition_code_period_ms(signal_type: type[Signal], signal: Signal) -> float:
    policy = ACQUISITION_POLICIES[signal_type.signal_type_id]
    component = signal.code_set.components[
        signal.code_set.index_of(policy.component_name)
    ]
    rate = signal_type.tracking_code_rate_chips_per_sec
    return len(acquisition_code(signal_type, signal)) / rate * 1e3


def acquisition_resolves_overlay_phase(
    signal_type: type[Signal], signal: Signal
) -> bool:
    """
    True when acquisition correlates a code spanning a whole overlay period, so
    the recovered code phase already carries the overlay index and no separate
    search is needed.  L5 acquires on Q x NH20, whose period is the counter's
    modulus, so it pins both NH20 and -- since the counters share a clock --
    NH10 as well.
    """
    period = signal.overlay_period_ms
    return period > 0 and acquisition_code_period_ms(signal_type, signal) >= period


def build_ambiguity_search(
    signal_type: type[Signal], signal: Signal
) -> AmbiguitySearchPlan | None:
    """
    The long-code search implied by `AcquisitionPolicy.ambiguous_component`, or
    None when acquisition locates the whole signal.

    The acquisition component plays no part in the search: it gave the code
    phase and Doppler, and the hypothesis stride is a whole acquisition period,
    so it would score identically under every hypothesis.  The search therefore
    correlates the ambiguous component alone, still on its own chips of the
    combined clock -- a partial view of the signal, which is what
    `allow_partial_coverage` is for.
    """
    policy = ACQUISITION_POLICIES[signal_type.signal_type_id]
    if policy.ambiguous_component is None:
        return None

    code_set = signal.code_set
    acquired = code_set.components[code_set.index_of(policy.component_name)]
    ambiguous = code_set.components[code_set.index_of(policy.ambiguous_component)]

    stride_chips = acquired.code_length
    num_hypotheses, remainder = divmod(ambiguous.code_length, stride_chips)
    if remainder or num_hypotheses < 2:
        raise ValueError(
            f"{signal_type.signal_type_id}: {policy.ambiguous_component!r} spans "
            f"{ambiguous.code_length} chips, which is not a whole multiple "
            f"(>1) of {policy.component_name!r}'s {stride_chips}"
        )

    rate = signal_type.tracking_code_rate_chips_per_sec
    period_sec = ambiguous.code_length / rate
    return AmbiguitySearchPlan(
        code_set=build_code_set([ambiguous], allow_partial_coverage=True),
        scored_component=0,
        num_hypotheses=int(num_hypotheses),
        hypothesis_stride_chips=float(stride_chips),
        code_rate_chips_per_sec=rate,
        description=f"{ambiguous.name} block within the {period_sec:g} s pilot code",
    )


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
    params: dict[str, bpsk_acquisition.AcqSignalCodeParameters] = {}
    for signal_id, signal in signals.items():
        # Already zero-filled on any sibling's chips, so it cannot correlate
        # against that sibling's code -- nothing for acquisition to special-case.
        sequence = acquisition_code(signal_type, signal)
        params[signal_id] = bpsk_acquisition.AcqSignalCodeParameters(
            rate_chips_per_sec=signal_type.tracking_code_rate_chips_per_sec,
            length_chips=len(sequence),
            sequence=sequence,
        )
    return params


def resolve_acquisition_ambiguities(
    signal_type: type[Signal],
    signals: dict[str, Signal],
    acquisition_results: dict[str, bpsk_acquisition.AcquisitionResult],
    sample_block: np.ndarray,
    acq_config: bpsk_acquisition.AcquisitionConfiguration,
    signal_ids: Iterable[str] | None = None,
) -> dict[str, ambiguity_resolution.AmbiguityResolution]:
    """
    Run each signal's long-code search over the acquisition dwell.

    The same samples acquisition used are reused, with the same coherent block
    structure, so the hypothesis scores carry exactly the Doppler loss the
    acquisition already survived and no refinement step is needed in between.

    Signals with no ambiguous component (L1 C/A, L5) are skipped, as are ones
    that did not acquire.
    """
    if signal_ids is None:
        signal_ids = [
            sid for sid, result in acquisition_results.items() if result.signal_detected
        ]

    resolutions: dict[str, ambiguity_resolution.AmbiguityResolution] = {}
    for signal_id in signal_ids:
        signal = signals[signal_id]
        plan = build_ambiguity_search(signal_type, signal)
        if plan is None:
            continue
        acq_result = acquisition_results[signal_id]
        resolutions[signal_id] = ambiguity_resolution.resolve_code_ambiguity(
            sample_block,
            acq_config.sample_rate,
            code_set=plan.code_set,
            scored_component=plan.scored_component,
            acquired_code_phase_sec=acq_result.acq_code_phase_seconds,
            doppler_hz=acq_result.acq_doppler_hz,
            carrier_freq_hz=signal_type.carrier_freq_hz,
            nominal_code_rate_chips_per_sec=plan.code_rate_chips_per_sec,
            coherent_length_samples=acq_config.coherent_length_samples,
            num_blocks=acq_config.num_blocks,
            num_hypotheses=plan.num_hypotheses,
            hypothesis_stride_chips=plan.hypothesis_stride_chips,
        )
    return resolutions


def create_tracking_channels(
    signal_type: type[Signal],
    signals: dict[str, Signal],
    acquisition_results: dict[str, bpsk_acquisition.AcquisitionResult],
    tracking_signal_ids: Iterable[str],
    loop_params: tracking_channel.TrackingLoopParameters,
    output_capacity: int,
    start_mode_pll: bool = False,
    ambiguity_resolutions: dict[str, ambiguity_resolution.AmbiguityResolution]
    | None = None,
) -> dict[str, TrackingChannelAdapter]:
    tracking_policy = TRACKING_POLICIES[signal_type.signal_type_id]
    channels: dict[str, TrackingChannelAdapter] = {}
    for signal_id in tracking_signal_ids:
        signal = signals[signal_id]
        acq_result = acquisition_results[signal_id]

        # The resolved offset is simply part of the code phase: it says how far into
        # the long code the signal actually is, which acquisition could only pin down
        # modulo the code it correlated against.  L2C needs it or CL is generated as
        # though at its code origin, right 1 time in 75.
        #
        # An unresolved (low-confidence) search is ignored rather than trusted,
        # leaving the pre-existing behaviour.
        code_phase_offset_ms = 0.0
        resolution_applied = False
        resolution = (ambiguity_resolutions or {}).get(signal_id)
        plan = build_ambiguity_search(signal_type, signal)
        if resolution is not None and resolution.resolved and plan is not None:
            resolution_applied = True
            code_phase_offset_ms = (
                resolution.offset_chips / plan.code_rate_chips_per_sec * 1e3
            )
        elif resolution is not None and plan is not None:
            # The search ran and could not decide.  Guessing would be worse than
            # waiting: a mis-phased wipe-off never locks at all once the pilot
            # discriminator drops Costas wrapping.  Say so, because the visible
            # consequence is that the configured integration does not take effect.
            warnings.warn(
                f"{signal_id}: {plan.description} unresolved "
                f"(confidence {resolution.confidence:.2f} < threshold); tracking starts at "
                f"{loop_params.coherent_integration_ms} ms and "
                f"{tracking_policy.synced_coherent_integration_ms} ms coherent integration is NOT "
                f"active. The channel falls back to searching for the overlay phase after "
                f"PLL lock, and will extend only if that succeeds.",
                RuntimeWarning,
                stacklevel=2,
            )

        code_rate_ms_per_sec = (
            1.0 + acq_result.acq_doppler_hz / signal_type.carrier_freq_hz
        ) * 1e3
        initial_state = tracking_channel.TrackingSignalState(
            uptime_epoch_ms=acq_result.uptime_epoch_ms,
            code_phase_ms=acq_result.acq_code_phase_seconds * 1e3 + code_phase_offset_ms,
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
        # A signal with a tiered code can start already synced: the acquisition
        # code phase is in the overlay's frame, so the overlay index of the first
        # correlation interval is just its integer millisecond.  That skips the
        # post-lock search entirely -- wipe-off and the pilot discriminator are
        # live from the first interval, while the coherent accumulation still
        # waits for PLL lock (see TrackingChannel).
        initial_overlay_counter = None
        if acquisition_resolves_overlay_phase(signal_type, signal):
            # The first correlation interval starts at ceil(code phase), so its
            # overlay index is that integer millisecond.
            initial_overlay_counter = int(
                np.ceil(initial_state.code_phase_ms)
            ) % signal.overlay_period_ms

        # A resolved ambiguity makes a component usable that was not before, so it
        # can change which components drive the loops -- for L2C that is the whole
        # point, moving the carrier onto the CL pilot.
        discriminator_policy = tracking_policy.discriminator_policy
        if resolution_applied and tracking_policy.resolved_discriminator_policy is not None:
            discriminator_policy = tracking_policy.resolved_discriminator_policy

        channel = tracking_channel.TrackingChannel(
            loop_params=loop_params,
            signal_params=signal_params,
            initial_signal_state=initial_state,
            output_capacity=output_capacity,
            discriminator_policy=discriminator_policy,
            synced_policy=tracking_policy.synced_discriminator_policy,
            synced_coherent_integration_ms=tracking_policy.synced_coherent_integration_ms,
            initial_overlay_counter=initial_overlay_counter,
        )

        adapter = TrackingChannelAdapter(signal=signal, channel=channel)
        if start_mode_pll:
            adapter.set_mode_pll()
        channels[signal_id] = adapter
    return channels
