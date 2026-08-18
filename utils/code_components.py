"""
Signal-agnostic description of a GNSS signal's spreading-code structure.

CHIPS
-----
A **chip** is one position on the signal's chip clock -- the clock running at the
signal's full chip rate: 1.023 Mcps for L1 C/A and for L2C's multiplexed CM/CL
stream, 10.23 Mcps for L5.  Code phase is measured on this axis, so the integer
part of a code phase *is* a chip index.

Every component's sequence is stated on that one axis.  Component `c` contributes

    sequence_c[k % code_length_c]

at chip `k` -- no per-component rate, no offset, no second kind of chip.  A
component that is not transmitting at a given chip carries 0 there, which the
correlator accumulates as nothing.  Time-division multiplexing is therefore data
rather than metadata: L2C's CM is (CM_0, 0, CM_1, 0, ...) and its CL is
(0, CL_0, 0, CL_1, ...), 20460 and 1534500 chips long respectively on the
1.023 Mcps combined clock.

    GPS L1 C/A   one component, every chip
    GPS L2C      CM/CL interleaved chip by chip (IS-GPS-200), zero-filled apart
    GPS L5       I/Q co-located on every chip, separated by carrier phase
    GPS L1C      D/P co-located on every chip, separated by carrier phase

Storing the zeros costs one byte per chip per component and buys a correlator
inner loop with no modulo and no divide in it -- measurably faster than deriving
presence from a rate and an offset, and it removes the only place the sequence
and its description could disagree.

BRANCHES
--------
Components that share a chip are separated by carrier phase instead, and
`branch` says which of the two quadrature carriers each one rides.  Nothing in
the correlator consumes it: correlating the complex baseband against each real
code already keeps them apart, and the 90 degrees lands in the carrier phase
estimate.  What `branch` records is the *relative* fact -- two components on the
same branch (L2C's CM and CL) can hand the carrier loop between them
phase-continuously, while two on opposite branches (L5's I and Q) cannot without
a quarter-cycle re-pull.

The absolute value is unobservable to a receiver tracking one signal in
isolation: L1 C/A and L2C both sit in quadrature to P(Y), which we never
process, so their `Branch.Q` changes no behaviour.  It is recorded because it is
true, and consumed only as a difference.

Note "slot" is deliberately avoided: in GNSS it already means an orbital slot or
a GLONASS frequency slot (see `GLONASS SLOT / FRQ #` in the RINEX reader).

NAMING
------
Identifiers say what a quantity *is* and carry its unit as a suffix
(`code_phase_chips`, `chips_per_sample`), matching `code_phase_ms` and
`code_rate_chips_per_sec` in the tracking channel.  "chips" is a unit, never a
name on its own.

There is only one chip axis now, so "chips" never needs qualifying.  What stays
qualified is the *per-component* array a code set flattens into
(`component_code_lengths`, `component_code_start_indices`), because those index
components rather than chips.

`subcarrier` and `power_weight` describe structure that later signals need
(BOC/TMBOC on L1C, L1C's 25/75 power split).  They are carried by the data model
from the outset so that adding those signals is a matter of configuration plus
one correlator capability, rather than a redesign.  Nothing consumes them yet.
`branch` is carried for the same reason, and is documented above.

This module is deliberately free of any GPS-specific constants; concrete signals are
assembled in `utils.signal_interfaces`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Branch(StrEnum):
    """
    Which of the two quadrature carriers a component rides.

    Per IS-GPS-200/705/800, relative to the P(Y) in-phase reference:

        GPS L1 C/A   Q
        GPS L2C      CM Q, CL Q   (same branch -- see the module docstring)
        GPS L5       I  I,  Q Q
        GPS L1C      D  I,  P Q

    Only differences between components of one signal are meaningful to a
    receiver; there is no default because the simplest signal in the catalog
    (L1 C/A) is on Q, so any default would be wrong somewhere obvious.
    """

    I = "I"
    Q = "Q"


class SubcarrierKind(StrEnum):
    BOC_SIN = "BOC_SIN"
    BOC_COS = "BOC_COS"
    TMBOC = "TMBOC"


@dataclass(frozen=True)
class Subcarrier:
    """
    Square-wave subcarrier applied on top of a component's spreading code.

    Not yet consumed by the correlator.  For TMBOC, `pattern` is a per-chip mask
    selecting which chips use `pattern_rate_hz` instead of `rate_hz` (GPS L1C-P uses
    BOC(6,1) on 4 of every 33 chips, BOC(1,1) on the rest).
    """

    kind: SubcarrierKind
    rate_hz: float
    pattern: np.ndarray | None = None
    pattern_rate_hz: float = 0.0

    def __post_init__(self) -> None:
        if self.rate_hz <= 0.0:
            raise ValueError("subcarrier rate_hz must be positive")
        if self.kind == SubcarrierKind.TMBOC:
            if self.pattern is None:
                raise ValueError("TMBOC requires a per-chip pattern")
            if self.pattern_rate_hz <= 0.0:
                raise ValueError("TMBOC requires a positive pattern_rate_hz")


@dataclass(frozen=True)
class CodeComponent:
    """
    One spreading code within a signal, stated on the signal's chip axis.

    `sequence` runs at the signal's full chip rate and carries 0 at every chip
    this component does not transmit on, so a time-multiplexed component is
    simply a zero-filled one -- see the module docstring.  There is no second
    kind of chip and no multiplexing metadata: what a component is, is its
    sequence and which carrier branch it rides.
    """

    name: str
    sequence: np.ndarray  # +/-1 where transmitting, 0 elsewhere; int8
    branch: Branch
    overlay: np.ndarray | None = None  # +/-1 tiered/secondary code
    subcarrier: Subcarrier | None = None
    power_weight: float = 1.0
    # Duration of one data symbol on this component; None for a dataless pilot.
    # This is what bounds coherent integration once any overlay is stripped: an
    # accumulation spanning a symbol boundary partly cancels itself.
    symbol_period_ms: int | None = None

    def __post_init__(self) -> None:
        if self.sequence.ndim != 1 or self.sequence.size == 0:
            raise ValueError(f"component {self.name!r}: sequence must be a non-empty 1-D array")
        if self.sequence.dtype != np.int8:
            raise ValueError(
                f"component {self.name!r}: sequence must be int8, got {self.sequence.dtype}. "
                "Some gnss_tools getters return float64 0/1 -- convert with "
                "(1 - 2 * seq).astype(np.int8)."
            )
        if not np.any(self.sequence):
            raise ValueError(
                f"component {self.name!r}: sequence is all zeros, so it transmits nothing"
            )
        if self.power_weight <= 0.0:
            raise ValueError(f"component {self.name!r}: power_weight must be positive")
        if self.symbol_period_ms is not None and self.symbol_period_ms < 1:
            raise ValueError(
                f"component {self.name!r}: symbol_period_ms must be >= 1 ms or None "
                f"(dataless), got {self.symbol_period_ms}"
            )

    @property
    def code_length(self) -> int:
        """Chips before this component's sequence repeats, zero chips included."""
        return int(self.sequence.size)


@dataclass(frozen=True)
class CodeSet:
    """
    A signal's components flattened into the arrays the numba correlator consumes.

    Codes have wildly different lengths (1023, 10230, 767250), so they are packed
    end to end into one contiguous int8 buffer rather than passed as a ragged
    sequence.  Component `c` occupies

        codes_flat[component_code_start_indices[c] : component_code_start_indices[c] + component_code_lengths[c]]

    `component_code_start_indices` is a *base address* into `codes_flat`, not a
    position on the chip axis.
    """

    components: tuple[CodeComponent, ...]
    codes_flat: np.ndarray  # int8, all component codes concatenated
    component_code_start_indices: np.ndarray  # int64, where each component's block starts in codes_flat
    component_code_lengths: np.ndarray  # int64, chips in each component's code
    pattern_period_chips: int  # chips before the whole pattern repeats

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components)

    @property
    def power_weights(self) -> np.ndarray:
        return np.array([c.power_weight for c in self.components], dtype=np.float64)

    @property
    def has_subcarrier(self) -> bool:
        return any(c.subcarrier is not None for c in self.components)

    @property
    def has_overlay(self) -> bool:
        return any(c.overlay is not None for c in self.components)

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no component named {name!r}; have {self.names}") from None

    @property
    def branches(self) -> tuple[Branch, ...]:
        return tuple(component.branch for component in self.components)

    def share_branch(self, *names: str) -> bool:
        """
        True when every named component rides the same carrier branch, i.e. the
        carrier loop can be handed between them without a quarter-cycle re-pull.

        L2C's CM and CL do; L5's I and Q do not.
        """
        return len({self.components[self.index_of(name)].branch for name in names}) == 1

    def wrap_code_phase_chips(self, code_phase_chips: float) -> float:
        """
        Reduce a code phase into one full repetition of the multiplexed pattern.

        Reducing by anything smaller -- e.g. one component's own length -- lands
        the components at different points in their own codes and silently
        misaligns them once the phase runs past a code period.
        """
        return float(code_phase_chips % self.pattern_period_chips)


def build_code_set(
    components: tuple[CodeComponent, ...] | list[CodeComponent],
    allow_partial_coverage: bool = False,
) -> CodeSet:
    """
    Flatten components into kernel-ready arrays and derive the pattern period.

    `allow_partial_coverage` waives the check that every chip carries at least
    one component.  That check exists to catch a *transmit* topology which would
    leave signal energy unclaimed, so it is right by default -- but a code set
    built to score one component in isolation (L2 CL alone, on odd chips)
    legitimately describes only part of the signal, and correlating it is
    exactly the intent.
    """
    components = tuple(components)
    if not components:
        raise ValueError("a signal needs at least one code component")

    names = [c.name for c in components]
    if len(set(names)) != len(names):
        raise ValueError(f"component names must be unique, got {names}")

    if not allow_partial_coverage:
        _validate_chip_coverage(components)

    codes_flat = np.concatenate([c.sequence for c in components]).astype(np.int8)
    code_lengths = np.array([c.code_length for c in components], dtype=np.int64)
    # Each block starts where the previous one ended.
    start_indices = np.zeros(len(components), dtype=np.int64)
    start_indices[1:] = np.cumsum(code_lengths)[:-1]

    # One repetition of the whole multiplexed pattern: every component must be back
    # at its own code origin simultaneously.
    pattern_period_chips = int(np.lcm.reduce(code_lengths))

    return CodeSet(
        components=components,
        codes_flat=codes_flat,
        component_code_start_indices=start_indices,
        component_code_lengths=code_lengths,
        pattern_period_chips=pattern_period_chips,
    )


def _validate_chip_coverage(components: tuple[CodeComponent, ...]) -> None:
    """
    Catch topologies that cannot be what the caller meant.

    Components may legitimately share a chip -- L5's I/Q and L1C's D/P are
    co-located and separated by carrier phase.  What is not legitimate is a chip
    that *no* component transmits on, because the correlator would silently skip
    signal energy there.  Passing L2C's CM without its CL is the case this
    catches: the odd chips would be zero in every component.

    This reads the sequences rather than a description of them, over one full
    repetition of the pattern.  Real spreading codes are +/-1, so a zero can only
    have come from the interleaving fill, which is what makes the test sound.
    """
    period = int(np.lcm.reduce([c.code_length for c in components]))
    covered = np.zeros(period, dtype=bool)
    for component in components:
        covered |= np.tile(component.sequence != 0, period // component.code_length)

    missing = np.flatnonzero(~covered)
    if missing.size:
        shown = ", ".join(str(chip) for chip in missing[:8])
        if missing.size > 8:
            shown += ", ..."
        raise ValueError(
            f"{missing.size} of {period} chips carry no component (chips {shown}); "
            "the correlator would skip signal energy there. If this code set "
            "deliberately describes only part of a signal, pass "
            "allow_partial_coverage=True."
        )


def epl_delay_bins(chip_spacing: float) -> tuple[float, ...]:
    """
    Classic early/prompt/late bin offsets in chips.

    Order is (early, prompt, late) -- early leads, so its offset is positive.
    BOC signals will need wider layouts (very-early/very-late) to resolve side-peak
    ambiguity, which is why delay bins are an explicit tuple rather than a count
    plus a step.
    """
    if chip_spacing <= 0.0:
        raise ValueError("chip_spacing must be positive")
    return (chip_spacing, 0.0, -chip_spacing)
