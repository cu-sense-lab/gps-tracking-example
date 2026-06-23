from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, Dict, Iterable, Protocol

import numpy as np

import gnss_tools.signals.gps_l1ca as gps_l1ca
import gnss_tools.signals.gps_l2c as gps_l2c

from . import bpsk_acquisition
from . import tracking_bpsk_aligned
from . import tracking_l2c_aligned


class SignalFamily(StrEnum):
    L1CA = "L1CA"
    L2C = "L2C"


class CorrelatorStrategyName(StrEnum):
    BPSK = "BPSK"
    INTERLEAVED_BPSK = "INTERLEAVED_BPSK"


@dataclass(frozen=True)
class SignalDefinition:
    signal_id: str
    family: SignalFamily
    correlator_strategy: CorrelatorStrategyName
    carrier_freq_hz: float
    acquisition_code_rate_chips_per_sec: float
    acquisition_code_length_chips: int
    acquisition_code_sequence: np.ndarray
    tracking_code_rate_chips_per_sec: float
    tracking_code_sequences: tuple[np.ndarray, ...]


class CorrelatorStrategy(Protocol):
    kind: CorrelatorStrategy

    def create_tracking_channel(
        self,
        signal_definition: SignalDefinition,
        loop_params: tracking_bpsk_aligned.TrackingLoopParameters,
        initial_signal_state: tracking_bpsk_aligned.TrackingSignalState,
        output_capacity: int,
    ) -> Any:
        ...


class BpskCorrelatorStrategy:
    kind = CorrelatorStrategy.BPSK

    def create_tracking_channel(
        self,
        signal_definition: SignalDefinition,
        loop_params: tracking_bpsk_aligned.TrackingLoopParameters,
        initial_signal_state: tracking_bpsk_aligned.TrackingSignalState,
        output_capacity: int,
    ) -> tracking_bpsk_aligned.TrackingChannel:
        signal_params = tracking_bpsk_aligned.TrackingSignalParameters(
            code_seq=signal_definition.tracking_code_sequences[0],
            nominal_code_rate_chips_per_sec=signal_definition.tracking_code_rate_chips_per_sec,
            carrier_freq_hz=signal_definition.carrier_freq_hz,
        )
        return tracking_bpsk_aligned.TrackingChannel(
            loop_params=loop_params,
            signal_params=signal_params,
            initial_signal_state=initial_signal_state,
            output_capacity=output_capacity,
        )


class InterleavedBpskCorrelatorStrategy:
    kind = CorrelatorStrategy.INTERLEAVED_BPSK

    def create_tracking_channel(
        self,
        signal_definition: SignalDefinition,
        loop_params: tracking_bpsk_aligned.TrackingLoopParameters,
        initial_signal_state: tracking_bpsk_aligned.TrackingSignalState,
        output_capacity: int,
    ) -> tracking_l2c_aligned.TrackingChannel:
        signal_params = tracking_l2c_aligned.L2CTrackingSignalParameters(
            code_seq_0=signal_definition.tracking_code_sequences[0],
            code_seq_1=signal_definition.tracking_code_sequences[1],
            nominal_code_rate_chips_per_sec=signal_definition.tracking_code_rate_chips_per_sec,
            carrier_freq_hz=signal_definition.carrier_freq_hz,
        )
        return tracking_l2c_aligned.TrackingChannel(
            loop_params=loop_params,
            signal_params=signal_params,
            initial_signal_state=initial_signal_state,
            output_capacity=output_capacity,
        )


STRATEGIES: Dict[CorrelatorStrategy, CorrelatorStrategy] = {
    CorrelatorStrategy.BPSK: BpskCorrelatorStrategy(),
    CorrelatorStrategy.INTERLEAVED_BPSK: InterleavedBpskCorrelatorStrategy(),
}


@dataclass
class TrackingChannelAdapter:
    signal_definition: SignalDefinition
    channel: Any

    @property
    def outputs(self) -> Any:
        return self.channel.outputs

    def process_sample_buffer(self, sample_buffer: Any) -> None:
        self.channel.process_sample_buffer(sample_buffer)

    def set_mode_pll(self) -> None:
        self.channel.loop_state.mode = tracking_bpsk_aligned.TrackingLoopMode.PLL

    def get_prompt_component(self, component: int = 0) -> np.ndarray:
        prompt = self.outputs.prompt_corr
        if getattr(prompt, "ndim", 1) == 1:
            return prompt
        return prompt[:, component]

    def get_early_component(self, component: int = 0) -> np.ndarray:
        early = self.outputs.early_corr
        if getattr(early, "ndim", 1) == 1:
            return early
        return early[:, component]

    def get_late_component(self, component: int = 0) -> np.ndarray:
        late = self.outputs.late_corr
        if getattr(late, "ndim", 1) == 1:
            return late
        return late[:, component]


def build_signal_definitions(
    family: SignalFamily,
    prns: Iterable[int] = range(1, 33),
) -> dict[str, SignalDefinition]:
    definitions: dict[str, SignalDefinition] = {}
    for prn in prns:
        signal_id = f"G{prn:02d}"
        if family == SignalFamily.L1CA:
            code_seq = 1 - 2 * gps_l1ca.get_GPS_L1CA_code_sequence(prn)
            definitions[signal_id] = SignalDefinition(
                signal_id=signal_id,
                family=family,
                correlator_strategy=CorrelatorStrategy.BPSK,
                carrier_freq_hz=gps_l1ca.CARRIER_FREQ,
                acquisition_code_rate_chips_per_sec=gps_l1ca.CODE_RATE,
                acquisition_code_length_chips=gps_l1ca.CODE_LENGTH,
                acquisition_code_sequence=code_seq,
                tracking_code_rate_chips_per_sec=gps_l1ca.CODE_RATE,
                tracking_code_sequences=(code_seq,),
            )
        elif family == SignalFamily.L2C:
            code_cm = 1 - 2 * gps_l2c.get_GPS_L2CM_code_sequence(prn)
            code_cl = 1 - 2 * gps_l2c.get_GPS_L2CL_code_sequence(prn)
            definitions[signal_id] = SignalDefinition(
                signal_id=signal_id,
                family=family,
                correlator_strategy=CorrelatorStrategy.INTERLEAVED_BPSK,
                carrier_freq_hz=gps_l2c.CARRIER_FREQ,
                acquisition_code_rate_chips_per_sec=gps_l2c.CODE_RATE_L2CM,
                acquisition_code_length_chips=gps_l2c.CODE_LENGTH_L2CM,
                acquisition_code_sequence=code_cm,
                tracking_code_rate_chips_per_sec=gps_l2c.CODE_RATE_L2CLM,
                tracking_code_sequences=(code_cm, code_cl),
            )
        else:
            raise ValueError(f"Unsupported family: {family}")
    return definitions


def build_acquisition_code_params(
    signal_definitions: dict[str, SignalDefinition],
) -> dict[str, bpsk_acquisition.AcqSignalCodeParameters]:
    return {
        signal_id: bpsk_acquisition.AcqSignalCodeParameters(
            rate_chips_per_sec=signal_def.acquisition_code_rate_chips_per_sec,
            length_chips=signal_def.acquisition_code_length_chips,
            sequence=signal_def.acquisition_code_sequence,
        )
        for signal_id, signal_def in signal_definitions.items()
    }


def create_tracking_channels(
    signal_definitions: dict[str, SignalDefinition],
    acquisition_results: dict[str, bpsk_acquisition.AcquisitionResult],
    tracking_signal_ids: Iterable[str],
    loop_params: tracking_bpsk_aligned.TrackingLoopParameters,
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
        initial_state = tracking_bpsk_aligned.TrackingSignalState(
            uptime_epoch_ms=acq_result.uptime_epoch_ms,
            code_phase_ms=acq_result.acq_code_phase_seconds * 1e3,
            code_rate_ms_per_sec=code_rate_ms_per_sec,
            carrier_phase_cycles=0.0,
            carrier_rate_cyc_per_sec=acq_result.acq_doppler_hz,
        )
        strategy = STRATEGIES[signal_def.correlator_strategy]
        channel = strategy.create_tracking_channel(
            signal_definition=signal_def,
            loop_params=loop_params,
            initial_signal_state=initial_state,
            output_capacity=output_capacity,
        )
        adapter = TrackingChannelAdapter(
            signal_definition=signal_def,
            channel=channel,
        )
        if start_mode_pll:
            adapter.set_mode_pll()
        channels[signal_id] = adapter
    return channels