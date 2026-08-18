"""
Tests for the signal-agnostic code topology model.

The critical property is `pattern_period_chips`: reducing a chip phase by anything
other than a full repetition of the multiplexed pattern misaligns the
`(k - component_offset_chips) // chips_per_component_chip` mapping once the phase
runs past a code period.  That is
the class of bug this model exists to make unrepresentable.
"""

from __future__ import annotations

import numpy as np
import pytest

import gnss_tools.signals.gps_l2c as gps_l2c

from utils.code_components import (
    BOCComponent,
    BPSKComponent,
    CodeComponent,
    QPSKComponent,
    Subcarrier,
    SubcarrierKind,
    TDBPSKComponent,
    build_code_set,
    epl_delay_bins,
)

from . import synthetic


def _code(length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (1 - 2 * rng.integers(0, 2, length)).astype(np.int8)


def test_single_component_period_is_code_length():
    code_set = build_code_set([BPSKComponent(name="CA", sequence=_code(1023))])
    assert code_set.pattern_period_chips == 1023
    assert code_set.num_components == 1


def test_interleaved_period_matches_hand_derived_value():
    """
    L2C: CM (10230) and CL (767250) interleaved on a 2x chip clock.  The pattern
    repeats every 2 * 767250 chips, since 10230 divides 767250.
    """
    cm, cl = synthetic.get_l2c_codes(1)
    code_set = build_code_set(
        [
            TDBPSKComponent(name="CM", sequence=cm, chips_per_component_chip=2, component_offset_chips=0),
            TDBPSKComponent(name="CL", sequence=cl, chips_per_component_chip=2, component_offset_chips=1),
        ]
    )
    assert code_set.pattern_period_chips == 2 * gps_l2c.CODE_LENGTH_L2CL
    assert code_set.pattern_period_chips == 1534500


def test_colocated_components_share_period():
    """L5 I/Q and L1C D/P sit at the same chips, separated by carrier phase."""
    code_set = build_code_set(
        [
            QPSKComponent(name="I", sequence=_code(10230, 1)),
            QPSKComponent(name="Q", sequence=_code(10230, 2)),
        ]
    )
    assert code_set.pattern_period_chips == 10230
    assert code_set.num_components == 2


def test_wrap_code_phase_preserves_component_alignment():
    """
    Wrapping by the pattern period must leave every component's derived code index
    unchanged.  Wrapping by max(len) instead -- the original bug -- does not.
    """
    cm, cl = synthetic.get_l2c_codes(1)
    code_set = build_code_set(
        [
            TDBPSKComponent(name="CM", sequence=cm, chips_per_component_chip=2, component_offset_chips=0),
            TDBPSKComponent(name="CL", sequence=cl, chips_per_component_chip=2, component_offset_chips=1),
        ]
    )
    period = code_set.pattern_period_chips
    for raw_phase in (0.0, 3.0, 20460.0, 1534499.0):
        wrapped = code_set.wrap_code_phase_chips(raw_phase + 5 * period)
        assert wrapped == pytest.approx(raw_phase % period)
        # same chip parity => same component is active
        assert int(wrapped) % 2 == int(raw_phase) % 2


def test_flat_packing_indexes_each_component_correctly():
    """
    `component_code_start_indices[c]` is a base address into `codes_flat`; a component's own
    chip index is a displacement from it.  The correlator adds the two, so this
    pins down that base + displacement recovers the original sequence.
    """
    a, b = _code(16, 1), _code(48, 2)
    code_set = build_code_set([QPSKComponent("A", a), QPSKComponent("B", b)])

    # Blocks must be laid end to end, otherwise the check below is vacuous.
    np.testing.assert_array_equal(code_set.component_code_start_indices, [0, len(a)])

    for i, expected in enumerate((a, b)):
        base = code_set.component_code_start_indices[i]
        length = code_set.component_code_lengths[i]
        np.testing.assert_array_equal(code_set.codes_flat[base : base + length], expected)
        # Element-wise, the form the kernel actually uses.
        for chip in range(length):
            assert code_set.codes_flat[base + chip] == expected[chip]


def test_index_of_and_metadata():
    code_set = build_code_set(
        [
            QPSKComponent("D", _code(64, 1), power_weight=0.25),
            QPSKComponent("P", _code(64, 2), power_weight=0.75),
        ]
    )
    assert code_set.index_of("P") == 1
    np.testing.assert_allclose(code_set.power_weights, [0.25, 0.75])
    assert not code_set.has_subcarrier
    assert not code_set.has_overlay
    with pytest.raises(KeyError):
        code_set.index_of("missing")


def test_overlay_and_subcarrier_flags():
    boc = Subcarrier(kind=SubcarrierKind.BOC_SIN, rate_hz=1.023e6)
    code_set = build_code_set(
        [
            QPSKComponent("I", _code(32, 1), overlay=_code(10, 3)),
            BOCComponent("Q", _code(32, 2), subcarrier=boc),
        ]
    )
    assert code_set.has_overlay
    assert code_set.has_subcarrier


def test_rejects_float_sequences():
    """gnss_tools' L5 getters return float64 0/1; catching that early is the point."""
    with pytest.raises(ValueError, match="int8"):
        BPSKComponent(name="I", sequence=np.zeros(10, dtype=np.float64))


def test_rejects_offset_outside_stride():
    with pytest.raises(ValueError, match="component_offset_chips"):
        TDBPSKComponent(name="X", sequence=_code(8), chips_per_component_chip=2, component_offset_chips=2)


def test_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        build_code_set([BPSKComponent("A", _code(8, 1)), BPSKComponent("A", _code(8, 2))])


def test_rejects_uncovered_signal_chips():
    """Two components both at offset 0, one chip apart, would skip every odd chip."""
    with pytest.raises(ValueError, match="claimed by no component"):
        build_code_set(
            [
                TDBPSKComponent("A", _code(8, 1), chips_per_component_chip=2, component_offset_chips=0),
                TDBPSKComponent("B", _code(8, 2), chips_per_component_chip=2, component_offset_chips=0),
            ]
        )


def test_rejects_direct_code_component_construction():
    """CodeComponent is abstract; every code must pick one of the four kinds."""
    with pytest.raises(TypeError, match="abstract"):
        CodeComponent(name="X", sequence=_code(8))


def test_bpsk_component_rejects_multiplexing_fields():
    with pytest.raises(ValueError, match="BPSKComponent"):
        BPSKComponent(name="X", sequence=_code(8), chips_per_component_chip=2)


def test_tdbpsk_component_requires_stride_of_at_least_two():
    with pytest.raises(ValueError, match="TDBPSKComponent"):
        TDBPSKComponent(name="X", sequence=_code(8))


def test_boc_component_requires_subcarrier():
    with pytest.raises(ValueError, match="subcarrier"):
        BOCComponent(name="X", sequence=_code(8))


def test_tmboc_requires_pattern():
    with pytest.raises(ValueError, match="TMBOC requires"):
        Subcarrier(kind=SubcarrierKind.TMBOC, rate_hz=1.023e6)


def test_epl_delay_bins_order_is_early_prompt_late():
    assert epl_delay_bins(0.5) == (0.5, 0.0, -0.5)
