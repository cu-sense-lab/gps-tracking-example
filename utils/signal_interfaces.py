from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

import numpy as np

import gnss_tools.signals.gps_l1ca as gps_l1ca
import gnss_tools.signals.gps_l2c as gps_l2c
import gnss_tools.signals.gps_l5 as gps_l5

from . import bpsk_acquisition
from . import sample_streaming
from . import tracking_channel
from .code_components import CodeComponent, CodeSet, build_code_set


class SignalFamily(StrEnum):
    L1CA = "L1CA"
    L2C = "L2C"
    L5 = "L5"


@dataclass(frozen=True)
class SignalDefinition:
    """
    Everything needed to acquire and track one signal from one satellite.

    Acquisition and tracking are described separately because they need not use the
    same code: L2C acquires on CM alone at 511.5 kcps, but tracks CM and CL together
    on the 1.023 Mcps interleaved stream.

    There is no longer a per-signal correlator strategy.  How a signal's components
    occupy the chip clock is data on the `CodeSet` (see `utils.code_components`),
    and which components drive which loop is data on the discriminator policy, so a
    single `TrackingChannel` serves every signal.
    """

    signal_id: str
    family: SignalFamily
    carrier_freq_hz: float

    # Acquisition
    acquisition_code_rate_chips_per_sec: float
    acquisition_code_length_chips: int
    acquisition_code_sequence: np.ndarray
    acquisition_is_interleaved: bool

    # Tracking
    code_set: CodeSet
    tracking_code_rate_chips_per_sec: float
    primary_period_ms: int
    discriminator_policy: tracking_channel.LoopDiscriminatorPolicy


    @property
    def component_names(self) -> tuple[str, ...]:
        return self.code_set.names


@dataclass
class TrackingChannelAdapter:
    """Pairs a tracking channel with the signal definition that configured it."""

    signal_definition: SignalDefinition
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
        return self.signal_definition.code_set.index_of(name)

    def get_prompt_component(self, component: int = 0) -> np.ndarray:
        return self.outputs.prompt_corr[:, component]

    def get_early_component(self, component: int = 0) -> np.ndarray:
        return self.outputs.early_corr[:, component]

    def get_late_component(self, component: int = 0) -> np.ndarray:
        return self.outputs.late_corr[:, component]


def _build_l1ca_definition(signal_id: str, prn: int) -> SignalDefinition:
    code = (1 - 2 * gps_l1ca.get_GPS_L1CA_code_sequence(prn)).astype(np.int8)
    return SignalDefinition(
        signal_id=signal_id,
        family=SignalFamily.L1CA,
        carrier_freq_hz=gps_l1ca.CARRIER_FREQ,
        acquisition_code_rate_chips_per_sec=gps_l1ca.CODE_RATE,
        acquisition_code_length_chips=gps_l1ca.CODE_LENGTH,
        acquisition_code_sequence=code,
        acquisition_is_interleaved=False,
        code_set=build_code_set([CodeComponent(name="CA", sequence=code)]),
        tracking_code_rate_chips_per_sec=gps_l1ca.CODE_RATE,
        primary_period_ms=1,
        discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=0, code_components=(0,), costas=True
        ),
    )


def _build_l2c_definition(signal_id: str, prn: int) -> SignalDefinition:
    code_cm = (1 - 2 * gps_l2c.get_GPS_L2CM_code_sequence(prn)).astype(np.int8)
    code_cl = (1 - 2 * gps_l2c.get_GPS_L2CL_code_sequence(prn)).astype(np.int8)
    # CM and CL alternate chips on the 1.023 Mcps combined clock, so each
    # contributes one component chip every two chips.
    code_set = build_code_set(
        [
            CodeComponent(name="CM", sequence=code_cm, chips_per_component_chip=2, component_offset_chips=0),
            CodeComponent(name="CL", sequence=code_cl, chips_per_component_chip=2, component_offset_chips=1),
        ]
    )
    return SignalDefinition(
        signal_id=signal_id,
        family=SignalFamily.L2C,
        carrier_freq_hz=gps_l2c.CARRIER_FREQ,
        acquisition_code_rate_chips_per_sec=gps_l2c.CODE_RATE_L2CM,
        acquisition_code_length_chips=gps_l2c.CODE_LENGTH_L2CM,
        acquisition_code_sequence=code_cm,
        acquisition_is_interleaved=True,
        code_set=code_set,
        tracking_code_rate_chips_per_sec=gps_l2c.CODE_RATE_L2CLM,
        # CM repeats every 20 ms; CL every 1.5 s.
        primary_period_ms=20,
        # Only CM drives the loops.  Adding CL to code_components would combine both
        # components non-coherently in the DLL -- a genuine improvement, but a
        # behavioural change that belongs in its own commit rather than riding along
        # with the channel unification.
        discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=0, code_components=(0,), costas=True
        ),
    )


def _build_l5_definition(signal_id: str, prn: int) -> SignalDefinition:
    # NOTE: unlike the L1CA/L2C getters, these return float64 0/1 rather than int8.
    code_i = (1 - 2 * gps_l5.get_GPS_L5I_code_sequence(prn)).astype(np.int8)
    code_q = (1 - 2 * gps_l5.get_GPS_L5Q_code_sequence(prn)).astype(np.int8)
    # I and Q share every chip -- they are separated by carrier phase, not
    # by time -- so both take one component chip per chip, at offset 0.
    # Correlating the complex baseband against each real code independently keeps
    # them apart; the inherent 90 degree relationship needs no special handling.
    code_set = build_code_set(
        [
            CodeComponent(name="I", sequence=code_i),
            CodeComponent(name="Q", sequence=code_q),
        ]
    )
    return SignalDefinition(
        signal_id=signal_id,
        family=SignalFamily.L5,
        carrier_freq_hz=gps_l5.CARRIER_FREQ,
        # Acquire on the Q (pilot) component.  At 1 ms coherent integration the two
        # perform identically -- Q's Neuman-Hofman overlay flips sign every 1 ms
        # just as I's CNAV symbols do -- but acquiring on the component that will
        # ultimately drive the loops keeps the code phase reference consistent.
        acquisition_code_rate_chips_per_sec=gps_l5.CODE_RATE,
        acquisition_code_length_chips=gps_l5.PRIMARY_CODE_LENGTH,
        acquisition_code_sequence=code_q,
        acquisition_is_interleaved=False,
        code_set=code_set,
        tracking_code_rate_chips_per_sec=gps_l5.CODE_RATE,
        # 10230 chips at 10.23 Mcps.  Exactly the interval granularity the aligned
        # correlator already uses, which is what makes tiered-code wipe-off cheap
        # to add later.
        primary_period_ms=1,
        # Carrier runs on Q, the pilot.  The plan had this on I pre-sync, but
        # starting on Q avoids a 90 degree phase discontinuity when Neuman-Hofman
        # sync later switches the loops to the pilot -- I and Q are in quadrature,
        # so swapping the carrier component mid-track would force the PLL to
        # re-pull.  At 1 ms the two are equally noisy, so there is no cost.
        #
        # Costas stays True until the overlay is stripped: NH20 flips Q's sign
        # every 1 ms, so the pilot is not yet effectively dataless.  Switching to
        # a full atan2 discriminator is what earns L5 its ~6 dB, and that arrives
        # with the tiered-code layer.
        #
        # The delay discriminator combines I and Q non-coherently.  They are equal
        # power, so this is worth ~3 dB and costs nothing: overlay and data flips
        # cancel in the magnitudes.
        discriminator_policy=tracking_channel.LoopDiscriminatorPolicy(
            carrier_component=1, code_components=(0, 1), costas=True
        ),
    )


_DEFINITION_BUILDERS = {
    SignalFamily.L1CA: _build_l1ca_definition,
    SignalFamily.L2C: _build_l2c_definition,
    SignalFamily.L5: _build_l5_definition,
}


def build_signal_definitions(
    family: SignalFamily,
    prns: Iterable[int] = range(1, 33),
) -> dict[str, SignalDefinition]:
    try:
        builder = _DEFINITION_BUILDERS[family]
    except KeyError:
        raise ValueError(f"Unsupported family: {family}") from None
    return {f"G{prn:02d}": builder(f"G{prn:02d}", prn) for prn in prns}


def build_acquisition_code_params(
    signal_definitions: dict[str, SignalDefinition],
) -> dict[str, bpsk_acquisition.AcqSignalCodeParameters]:
    return {
        signal_id: bpsk_acquisition.AcqSignalCodeParameters(
            rate_chips_per_sec=signal_def.acquisition_code_rate_chips_per_sec,
            length_chips=signal_def.acquisition_code_length_chips,
            sequence=signal_def.acquisition_code_sequence,
            is_interleaved=signal_def.acquisition_is_interleaved,
        )
        for signal_id, signal_def in signal_definitions.items()
    }


def create_tracking_channels(
    signal_definitions: dict[str, SignalDefinition],
    acquisition_results: dict[str, bpsk_acquisition.AcquisitionResult],
    tracking_signal_ids: Iterable[str],
    loop_params: tracking_channel.TrackingLoopParameters,
    output_capacity: int,
    start_mode_pll: bool = False,
) -> dict[str, TrackingChannelAdapter]:
    channels: dict[str, TrackingChannelAdapter] = {}
    for signal_id in tracking_signal_ids:
        signal_def = signal_definitions[signal_id]
        acq_result = acquisition_results[signal_id]

        code_rate_ms_per_sec = (
            1.0 + acq_result.acq_doppler_hz / signal_def.carrier_freq_hz
        ) * 1e3
        initial_state = tracking_channel.TrackingSignalState(
            uptime_epoch_ms=acq_result.uptime_epoch_ms,
            code_phase_ms=acq_result.acq_code_phase_seconds * 1e3,
            code_rate_ms_per_sec=code_rate_ms_per_sec,
            carrier_phase_cycles=0.0,
            carrier_rate_cyc_per_sec=acq_result.acq_doppler_hz,
        )
        signal_params = tracking_channel.TrackingSignalParameters(
            code_set=signal_def.code_set,
            nominal_code_rate_chips_per_sec=signal_def.tracking_code_rate_chips_per_sec,
            carrier_freq_hz=signal_def.carrier_freq_hz,
            primary_period_ms=signal_def.primary_period_ms,
        )
        channel = tracking_channel.TrackingChannel(
            loop_params=loop_params,
            signal_params=signal_params,
            initial_signal_state=initial_state,
            output_capacity=output_capacity,
            discriminator_policy=signal_def.discriminator_policy,
        )

        adapter = TrackingChannelAdapter(signal_definition=signal_def, channel=channel)
        if start_mode_pll:
            adapter.set_mode_pll()
        channels[signal_id] = adapter
    return channels
