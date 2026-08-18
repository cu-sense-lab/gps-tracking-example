"""
Signal-agnostic description of a GNSS signal's spreading-code structure.

CHIP vs COMPONENT CHIP
----------------------
A **chip**, unqualified, is one position on the signal's chip clock -- the clock
running at the signal's full chip rate: 1.023 Mcps for L1 C/A and for L2C's
multiplexed CM/CL stream, 10.23 Mcps for L5.  Code phase is measured on this
axis, so the integer part of a code phase *is* a chip index.

A **component chip** is one position within a single component's own code
sequence.  A component's chips can be sparser than the signal's: L2 CM chips run
at 511.5 kcps on a 1.023 Mcps signal.

The two coincide only for a single-code signal.  How they relate is the
multiplexing scheme:

    L1 C/A   one component, every chip         chip n     -> CA component chip n
    L2C      CM and CL alternate (IS-GPS-200   chip 2n    -> CM component chip n
             chip-by-chip time multiplexing)   chip 2n+1  -> CL component chip n
    L5       I and Q share every chip,         chip n     -> I component chip n
             separated by carrier phase                   -> and Q component chip n

Component `c` is present at chip `k` when

    (k - component_offset_chips_c) % chips_per_component_chip_c == 0

and, when present, contributes

    sequence_c[((k - component_offset_chips_c) // chips_per_component_chip_c) % code_length_c]

`chips_per_component_chip_c` is how many chips pass between successive chips of
that component -- a recurrence interval, not the width of a chip.  (In L2C each
CM chip is transmitted as a one-chip pulse; what takes two is the gap until the
next CM chip, which is why CM's own rate is half the signal's.)
`component_offset_chips_c` is which of those chips it takes -- 0 for L2 CM,
1 for L2 CL.

That one rule covers every multiplexing scheme in the repository:

    GPS L1 C/A   one component,  1 chip/component chip                (1.023 Mcps)
    GPS L2C      CM/CL,          2 chips/component chip, offsets 0,1  (1.023 Mcps combined)
    GPS L5       I/Q,            1 chip/component chip, both present  (10.23 Mcps)
    GPS L1C      D/P,            1 chip/component chip, both present  (1.023 Mcps)

L5 and L1C place their two components in carrier quadrature rather than in
separate chips, so both are "always present" and are separated by correlating the
same complex baseband against each real code independently -- no quadrature
handling is needed here or in the correlator.

Note "slot" is deliberately avoided: in GNSS it already means an orbital slot or
a GLONASS frequency slot (see `GLONASS SLOT / FRQ #` in the RINEX reader).

NAMING
------
Identifiers say what a quantity *is* and carry its unit as a suffix
(`code_phase_chips`, `chips_per_sample`), matching `code_phase_ms` and
`code_rate_chips_per_sec` in the tracking channel.  "chips" is a unit, never a
name on its own.

Bare "chips" means the signal's chips, since that is the axis code phase lives
on.  Only component-level quantities are qualified (`component_code_lengths`,
`component_chip_index`) -- and inside `CodeComponent` itself even that is dropped,
because everything there is about that one component.

`overlay`, `subcarrier` and `power_weight` describe structure that later signals
need (tiered codes on L5/L1C, BOC/TMBOC on L1C, L1C's 25/75 power split).  They are
carried by the data model from the outset so that adding those signals is a matter
of configuration plus one correlator capability, rather than a redesign.  Nothing
consumes them yet.

This module is deliberately free of any GPS-specific constants; concrete signals are
assembled in `utils.signal_interfaces`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


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
    One spreading code within a signal, plus how it sits on the chip axis.

    Inside this class bare "chips" means this component's own chips.

    Abstract base -- construct one of the four kinds below instead (see
    TODO_SIGNALS.md): `BPSKComponent` (a single code, present at every chip),
    `QPSKComponent` (one leg of a carrier-quadrature pair, e.g. L5 I/Q),
    `TDBPSKComponent` (time-division multiplexed, e.g. L2C's CM/CL), or
    `BOCComponent` (carries a subcarrier). All four share this same field
    shape -- `build_code_set` and the correlator only ever see the base
    shape -- so each subclass exists purely to name and validate one
    multiplexing scheme; the class itself is what "type" a component is.
    """

    name: str
    sequence: np.ndarray  # +/-1, int8
    # Chips between successive component chips: 2 for L2C's CM and CL (each
    # contributes one component chip every other chip), 1 for a single-code
    # signal.  `component_offset_chips` picks which of those chips it takes.
    chips_per_component_chip: int = 1
    component_offset_chips: int = 0
    overlay: np.ndarray | None = None  # +/-1 tiered/secondary code
    subcarrier: Subcarrier | None = None
    power_weight: float = 1.0

    def __post_init__(self) -> None:
        if type(self) is CodeComponent:
            raise TypeError(
                "CodeComponent is abstract; construct a BPSKComponent, QPSKComponent, "
                "TDBPSKComponent, or BOCComponent instead"
            )
        if self.sequence.ndim != 1 or self.sequence.size == 0:
            raise ValueError(f"component {self.name!r}: sequence must be a non-empty 1-D array")
        if self.sequence.dtype != np.int8:
            raise ValueError(
                f"component {self.name!r}: sequence must be int8, got {self.sequence.dtype}. "
                "Some gnss_tools getters return float64 0/1 -- convert with "
                "(1 - 2 * seq).astype(np.int8)."
            )
        if self.chips_per_component_chip < 1:
            raise ValueError(f"component {self.name!r}: chips_per_component_chip must be >= 1")
        if not (0 <= self.component_offset_chips < self.chips_per_component_chip):
            raise ValueError(
                f"component {self.name!r}: component_offset_chips must satisfy "
                f"0 <= component_offset_chips < chips_per_component_chip, got {self.component_offset_chips} "
                f"with chips_per_component_chip {self.chips_per_component_chip}"
            )
        if self.power_weight <= 0.0:
            raise ValueError(f"component {self.name!r}: power_weight must be positive")

    @property
    def code_length(self) -> int:
        return int(self.sequence.size)

    @property
    def code_span_chips(self) -> int:
        """
        Chips covered by one full pass through this component's code.

        For L2 CM: 2 x 10230 = 20460 chips = 20 ms, even though the code itself
        is only 10230 component chips long.
        """
        return self.chips_per_component_chip * self.code_length


@dataclass(frozen=True)
class BPSKComponent(CodeComponent):
    """A single spreading code, present at every chip -- e.g. GPS L1 C/A's CA code."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chips_per_component_chip != 1 or self.component_offset_chips != 0:
            raise ValueError(
                f"component {self.name!r}: BPSKComponent is a single code at every chip; "
                "use TDBPSKComponent for a time-multiplexed one"
            )
        if self.subcarrier is not None:
            raise ValueError(f"component {self.name!r}: BPSKComponent has no subcarrier; use BOCComponent")


@dataclass(frozen=True)
class QPSKComponent(CodeComponent):
    """
    One leg of a carrier-quadrature pair sharing every chip -- e.g. GPS L5's I/Q,
    GPS L1C's D/P. Positionally identical to BPSKComponent (both are present at
    every chip); the separate class only tags "this is one leg of an I/Q pair",
    which the correlator does not need to treat specially -- correlating the
    complex baseband against each real code independently already keeps them
    apart.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chips_per_component_chip != 1 or self.component_offset_chips != 0:
            raise ValueError(
                f"component {self.name!r}: QPSKComponent sits at every chip like its "
                "quadrature sibling; use TDBPSKComponent for a time-multiplexed one"
            )
        if self.subcarrier is not None:
            raise ValueError(f"component {self.name!r}: QPSKComponent has no subcarrier; use BOCComponent")


@dataclass(frozen=True)
class TDBPSKComponent(CodeComponent):
    """Time-division multiplexed BPSK component -- e.g. GPS L2C's CM and CL."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chips_per_component_chip < 2:
            raise ValueError(
                f"component {self.name!r}: TDBPSKComponent needs chips_per_component_chip >= 2; "
                "a single-slot code is a BPSKComponent"
            )
        if self.subcarrier is not None:
            raise ValueError(f"component {self.name!r}: TDBPSKComponent has no subcarrier; use BOCComponent")


@dataclass(frozen=True)
class BOCComponent(CodeComponent):
    """
    Component carrying a square-wave subcarrier -- e.g. future GPS L1C
    (TMBOC), Galileo E1/E5 (CBOC). Not yet consumed by the correlator (see
    `correlate__multicomponent`); this class exists so the shape is ready.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.subcarrier is None:
            raise ValueError(f"component {self.name!r}: BOCComponent requires a subcarrier")


@dataclass(frozen=True)
class CodeSet:
    """
    A signal's components flattened into the arrays the numba correlator consumes.

    Codes have wildly different lengths (1023, 10230, 767250), so they are packed
    end to end into one contiguous int8 buffer rather than passed as a ragged
    sequence.  Component `c` occupies

        codes_flat[component_code_start_indices[c] : component_code_start_indices[c] + component_code_lengths[c]]

    `component_code_start_indices` is a *base address* into `codes_flat`;
    `component_offset_chips` is a *position on the chip axis*.  Unrelated
    quantities, so they are named apart rather than both called "offset".
    """

    components: tuple[CodeComponent, ...]
    codes_flat: np.ndarray  # int8, all component codes concatenated
    component_code_start_indices: np.ndarray  # int64, where each component's block starts in codes_flat
    component_code_lengths: np.ndarray  # int64, component chips in each code
    chips_per_component_chip: np.ndarray  # int64, chips between successive component chips
    component_offset_chips: np.ndarray  # int64, which of those chips each component takes
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

    def wrap_code_phase_chips(self, code_phase_chips: float) -> float:
        """
        Reduce a code phase into one full repetition of the multiplexed pattern.

        Reducing by anything smaller -- e.g. one component's own length -- shifts
        the `(k - component_offset_chips) // chips_per_component_chip` mapping and silently misaligns
        every component once the phase runs past a code period.
        """
        return float(code_phase_chips % self.pattern_period_chips)


def build_code_set(components: tuple[CodeComponent, ...] | list[CodeComponent]) -> CodeSet:
    """Flatten components into kernel-ready arrays and derive the pattern period."""
    components = tuple(components)
    if not components:
        raise ValueError("a signal needs at least one code component")

    names = [c.name for c in components]
    if len(set(names)) != len(names):
        raise ValueError(f"component names must be unique, got {names}")

    _validate_chip_coverage(components)

    codes_flat = np.concatenate([c.sequence for c in components]).astype(np.int8)
    code_lengths = np.array([c.code_length for c in components], dtype=np.int64)
    # Each block starts where the previous one ended.
    start_indices = np.zeros(len(components), dtype=np.int64)
    start_indices[1:] = np.cumsum(code_lengths)[:-1]

    # One repetition of the whole multiplexed pattern: every component must be back
    # at its own code origin simultaneously.
    pattern_period_chips = int(np.lcm.reduce([c.code_span_chips for c in components]))

    return CodeSet(
        components=components,
        codes_flat=codes_flat,
        component_code_start_indices=start_indices,
        component_code_lengths=code_lengths,
        chips_per_component_chip=np.array([c.chips_per_component_chip for c in components], dtype=np.int64),
        component_offset_chips=np.array([c.component_offset_chips for c in components], dtype=np.int64),
        pattern_period_chips=pattern_period_chips,
    )


def _validate_chip_coverage(components: tuple[CodeComponent, ...]) -> None:
    """
    Catch topologies that cannot be what the caller meant.

    Two components sharing both `chips_per_component_chip` and
    `component_offset_chips` are co-located (L5 I/Q, L1C D/P) and are separated by
    carrier phase -- that is legitimate.  But a mix that leaves some chip
    claimed by no component means the correlator would silently skip signal
    energy there.
    """
    if {c.chips_per_component_chip for c in components} == {1}:
        return  # every component is present at every chip

    covered: set[int] = set()
    phase_cycle = int(np.lcm.reduce([c.chips_per_component_chip for c in components]))
    for component in components:
        for chip in range(phase_cycle):
            if (
                chip - component.component_offset_chips
            ) % component.chips_per_component_chip == 0:
                covered.add(chip)
    missing = sorted(set(range(phase_cycle)) - covered)
    if missing:
        raise ValueError(
            f"chips {missing} are claimed by no component "
            f"(chips_per_component_chip={[c.chips_per_component_chip for c in components]}, "
            f"component_offset_chips={[c.component_offset_chips for c in components]}); "
            "the correlator would skip signal energy there"
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
