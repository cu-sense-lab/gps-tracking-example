import numpy as np
import numba as nb
from numpy.typing import NDArray

from .code_components import CodeSet


@nb.jit(nopython=True, parallel=False)
def numba_correlate__multicomponent__complex64(
        samples: nb.complex64[:],  # type: ignore
        codes_flat: nb.int8[:],  # type: ignore
        component_code_start_indices: nb.int64[:],  # type: ignore
        component_code_lengths: nb.int64[:],  # type: ignore
        initial_code_phase_chips: nb.float64,  # type: ignore
        chips_per_sample: nb.float64,  # type: ignore
        bin_offsets_chips: nb.float64[:],  # type: ignore
        conj_carr_sample: nb.complex64,  # type: ignore
        conj_carr_rotation: nb.complex64,  # type: ignore
        corr_values: nb.complex64[:, :]  # type: ignore
    ) -> None:
    """
    Accumulate one correlation interval for every (delay bin, code component) pair.

    Every component is stated on the signal's own chip axis, so component `c`
    contributes `codes_flat[start[c] + chip_index % component_code_lengths[c]]` at
    chip `chip_index` -- no per-component rate or offset to apply.  A component
    that is not transmitting at that chip carries 0 there and accumulates
    nothing, which is how time-division multiplexing (L2C's CM/CL) is expressed;
    see `utils.code_components`.

    Codes are packed end to end in `codes_flat` because their lengths differ by
    three orders of magnitude (1023 vs 1534500).  Component `c`'s code lives at
    `codes_flat[component_code_start_indices[c] : component_code_start_indices[c] + component_code_lengths[c]]`.

    `component_code_start_indices` addresses `codes_flat`; `bin_offsets_chips` is
    a position on the chip axis.  Different quantities, named apart.

    `corr_values` is accumulated in place and is NOT zeroed here: a correlation
    interval may be spread across several sample buffers.
    """
    num_samples = len(samples)
    num_bins = len(bin_offsets_chips)
    num_components = len(component_code_start_indices)

    code_phase_chips = initial_code_phase_chips
    for i in range(num_samples):
        carrierless = samples[i] * conj_carr_sample
        for j in range(num_bins):
            # floor, not truncation: a late bin near a code wrap produces a
            # slightly negative chip index, which must map to the previous chip.
            chip_index = int(np.floor(code_phase_chips + bin_offsets_chips[j]))
            for c in range(num_components):
                symbol = codes_flat[
                    component_code_start_indices[c]
                    + chip_index % component_code_lengths[c]
                ]
                if symbol == 1:
                    corr_values[j, c] += carrierless
                elif symbol == -1:
                    corr_values[j, c] -= carrierless
                elif symbol != 0:
                    corr_values[j, c] += carrierless * symbol
        conj_carr_sample *= conj_carr_rotation
        code_phase_chips += chips_per_sample


def correlate__multicomponent(
    samples: np.ndarray,
    samp_rate: float,
    initial_carr_phase_cycles: float,
    doppler_freq_hz: float,
    code_set: CodeSet,
    code_rate_chips_per_sec: float,
    initial_code_phase_chips: float,
    bin_offsets_chips: NDArray[np.float64],
    output: NDArray[np.complex64],
) -> None:
    """
    Correlate `samples` against every component of `code_set`.

    `output` has shape (num_bins, num_components) and is accumulated in place.
    `code_rate_chips_per_sec` and `initial_code_phase_chips` are on the signal's
    combined chip clock -- 1.023 Mcps for L1 C/A and for L2C's interleaved CM/CL
    stream, 10.23 Mcps for L5.
    """
    if code_set.has_subcarrier:
        # Dispatching here (rather than branching per sample inside the kernel)
        # keeps subcarrier-free signals off that code path entirely.  The BOC/TMBOC
        # kernel lands with GPS L1C.
        raise NotImplementedError("subcarrier (BOC/TMBOC) correlation is not implemented yet")

    wrapped_code_phase_chips = code_set.wrap_code_phase_chips(initial_code_phase_chips)
    chips_per_sample = code_rate_chips_per_sec / samp_rate

    conj_carr_sample = np.exp(-2j * np.pi * initial_carr_phase_cycles)
    conj_carr_rotation = np.exp(-2j * np.pi * doppler_freq_hz / samp_rate)

    numba_correlate__multicomponent__complex64(
        samples,
        code_set.codes_flat,
        code_set.component_code_start_indices,
        code_set.component_code_lengths,
        wrapped_code_phase_chips,
        chips_per_sample,
        bin_offsets_chips,
        conj_carr_sample,
        conj_carr_rotation,
        output,
    )

