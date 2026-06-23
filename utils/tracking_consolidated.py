



from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from utils import sample_streaming
from utils.bpsk_correlation import correlate__delay

# All signal state instances should inherit from this class, and they all correspond to a particular uptime epoch

@dataclass
class TrackingSignalState(ABC):
    uptime_epoch_ms: float

    @abstractmethod
    def to_dict(self) -> Dict[str, float]:
        pass

@dataclass
class TrackingSignalState_CodeCarrier(TrackingSignalState):
    code_phase_ms: float
    carrier_phase_cycles: float
    code_rate_sec_per_sec: float
    carrier_rate_cycles_per_sec: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "code_phase_ms": self.code_phase_ms,
            "carrier_phase_cycles": self.carrier_phase_cycles,
            "code_rate_sec_per_sec": self.code_rate_sec_per_sec,
            "carrier_rate_cycles_per_sec": self.carrier_rate_cycles_per_sec
        }
    
    def propagate_phase(self, uptime_epoch_ms: float) -> Tuple[float, float]:
        """Propagate the code and carrier phase to a new uptime epoch, given the current rates.
        Returns: (new_code_phase_ms, new_carrier_phase_cycles)
        """
        delta_time_sec = (uptime_epoch_ms - self.uptime_epoch_ms) / 1000.0
        new_code_phase_ms = self.code_phase_ms + self.code_rate_sec_per_sec * delta_time_sec
        new_carrier_phase_cycles = self.carrier_phase_cycles + self.carrier_rate_cycles_per_sec * delta_time_sec
        return new_code_phase_ms, new_carrier_phase_cycles


# Correlators implement a CorrelationStrategy
# AlignedCorrelator has a strategy and a delay/doppler configuration and storage

@dataclass
class CorrelatorConfig:
    num_components: int
    num_delays: int
    delay_offset_chips: float
    delay_step_chips: float
    num_dopplers: int
    doppler_offset_hz: float
    doppler_step_hz: float


# correlator performs correlation for delays and dopplers and some number of signal components
# it has a strategy that actually implements the correlation part
# this correlator handles bounds checking and correlation interval timing corrections
# then the strategy performs the actual correlation in delay and doppler (and signal components)
# the storage is owned by the correlator and passed to the strategy
# the correlator/tracking channel is responsible for resetting the correlator storage at the appropriate times

class Correlator:
    """
    Request-driven correlator.

    The tracking channel owns all dynamic signal and epoch state. The correlator
    owns only static configuration and performs in-place accumulation for the
    interval that the channel requests.
    """

    def __init__(
        self,
        config: CorrelatorConfig,
    ):
        self.config = config
        self.corr_grid = np.zeros((config.num_components, config.num_delays, config.num_dopplers), dtype=np.complex64)
        # We could approximate a single correlation count for all delay/doppler bins, which would be more efficient
        # For now, let's continue to use a corr_count grid, and in the future it won't be too hard to change.
        self.corr_counts = np.zeros((config.num_components, config.num_delays, config.num_dopplers), dtype=int)

    def reset(self) -> None:
        self.corr_grid.fill(0.0)
        self.corr_counts.fill(0)

    def accumulate(
        self,
        buffer: sample_streaming.SampleBuffer,
        accum_start_uptime_ms: float,
        accum_stop_uptime_ms: float,
        signal_params: TrackingSignalParameters,
        signal_state: TrackingSignalState,
    ) -> None:
        
        # first, need to determine start and end samples
        accum_start_sample_index = int((accum_start_uptime_ms - buffer.start_uptime_ms) * buffer.samp_rate / 1000)
        accum_stop_sample_index = int((accum_stop_uptime_ms - buffer.start_uptime_ms) * buffer.samp_rate / 1000)
        if not (accum_start_sample_index < accum_stop_sample_index):
            raise ValueError("accumulation stop must be after accumulation start")

        if accum_start_sample_index < 0:
            accum_start_sample_index = 0
        if accum_stop_sample_index > len(buffer.samples):
            accum_stop_sample_index = len(buffer.samples)
        
        actual_accum_start_uptime_ms = buffer.start_uptime_ms + accum_start_sample_index / buffer.samp_rate * 1000
        samples = buffer.samples[accum_start_sample_index:accum_stop_sample_index]
        num_accum_samples = len(samples)
        if num_accum_samples == 0:
            # NOTE: this should not happen
            print(f"Warning: no samples to accumulate for interval {accum_start_uptime_ms} ms to {accum_stop_uptime_ms} ms (buffer from {buffer.start_uptime_ms} ms to {buffer.stop_uptime_ms} ms)")
            print(f"accum_start_sample_index: {accum_start_sample_index}, accum_stop_sample_index: {accum_stop_sample_index}")
            raise ValueError("no samples to accumulate")

        # propagation uses correlation doppler because we may be accumulating a partial interval
        # (in that case, we want to use the same doppler for propagation as we do for correlation)
        # note: code rate will be decoupled from doppler
        dt_sec = (actual_accum_start_uptime_ms - signal_state.uptime_epoch_ms) * 1e-3
        code_phase_ms = signal_state.code_phase_ms + dt_sec * signal_state.code_rate_ms_per_sec
        code_phase_chips = code_phase_ms * 1e-3 * signal_params.nominal_code_rate_chips_per_sec
        code_rate_chips_per_sec = signal_state.code_rate_ms_per_sec * 1e-3 * signal_params.nominal_code_rate_chips_per_sec
        for i_dopp in range(self.config.num_dopplers):
            corr_doppler_offset_hz = self.config.doppler_offset_hz + i_dopp * self.config.doppler_step_hz
            corr_doppler_hz = signal_state.carrier_rate_cyc_per_sec + corr_doppler_offset_hz
            corr_carrier_phase_cycles = signal_state.carrier_phase_cycles + dt_sec * corr_doppler_hz
            correlate__delay(
                samples,
                buffer.samp_rate,
                corr_carrier_phase_cycles,
                corr_doppler_hz,
                signal_params.code_seq,
                signal_params.code_length_chips,
                code_rate_chips_per_sec,
                code_phase_chips,
                self.config.num_delays,
                self.config.delay_offset_chips,
                self.config.delay_step_chips,
                self.corr_grid[:, i_dopp],
            )
            self.corr_counts[:, i_dopp] += num_accum_samples