"""
Golden regression: tracking output must not drift.

A failure here means numerical behaviour changed.  Investigate before regenerating
the goldens -- that is the whole point of the file.

The baseline was regenerated once, during the channel unification, for two
deliberate and verified changes:

  1. Replica chip indices are now floored rather than truncated toward zero.
     Correlation intervals are aligned to integer ms of code phase and L1 C/A's
     code period is 1 ms, so every interval begins at code phase ~ 0 chips and the late
     bin at -0.5 chips is negative *on every epoch*.  Truncation read chip 0
     instead of the previous period's last chip, biasing the late correlator and
     the DLL systematically.  See `test_aligned_epoch_late_bin_reads_the_previous_code_period`.

  2. The delay discriminator combines magnitudes in float64.  Previously
     `np.abs()` on a complex64 correlator returned float32, so the discriminator
     carried float32 precision into a float64 state update.

Both were validated against explicitly constructed replicas in
`tests/test_correlator.py` before the baseline was replaced.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from .driver import OUTPUT_FIELDS, run_scenario
from .scenarios import SCENARIOS, TrackingScenario

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"

# The unified correlator sums the same products in a different loop order, so
# floating-point association can differ in the last bits.  Tolerances are far
# tighter than any real behavioural change would produce.
RTOL = 1e-9
ATOL = 1e-9


def _load_golden(name: str):
    path = GOLDEN_DIR / f"{name}.npz"
    if not path.exists():
        pytest.fail(f"missing golden baseline {path}; run `python -m tests.generate_golden`")
    return np.load(path, allow_pickle=False)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_tracking_matches_golden(scenario: TrackingScenario) -> None:
    golden = _load_golden(scenario.name)
    actual = run_scenario(scenario)

    assert int(actual["output_index"]) == int(golden["output_index"]), "epoch count changed"
    assert str(actual["final_mode"]) == str(golden["final_mode"]), "final loop mode changed"

    for field in OUTPUT_FIELDS:
        expected_arr = golden[field]
        actual_arr = actual[field]
        assert actual_arr.shape == expected_arr.shape, f"{field}: shape changed"
        np.testing.assert_allclose(
            actual_arr, expected_arr, rtol=RTOL, atol=ATOL, err_msg=f"{field} drifted"
        )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_tracking_converges(scenario: TrackingScenario) -> None:
    """
    Behavioural floor, independent of the goldens.

    Guards against the goldens themselves being regenerated from broken code: these
    thresholds encode what "tracking works" means rather than what it used to output.
    """
    actual = run_scenario(scenario)

    final_doppler_error = float(actual["doppler_freq_hz"][-1]) - scenario.doppler_hz
    assert abs(final_doppler_error) < 5.0, f"Doppler did not converge: {final_doppler_error:+.2f} Hz"

    assert str(actual["final_mode"]) == "PLL", "loop never reached PLL lock"

    assert float(actual["prompt_corr_circ_length"][-1]) > 0.9, "prompt phase not coherent"

    # Prompt should dominate early/late for an aligned 0.5-chip EPL correlator.
    prompt = actual["prompt_corr"][:, 0]
    early = actual["early_corr"][:, 0]
    late = actual["late_corr"][:, 0]
    tail = slice(-500, None)
    prompt_mag = np.abs(prompt[tail]).mean()
    assert prompt_mag > np.abs(early[tail]).mean(), "prompt not above early"
    assert prompt_mag > np.abs(late[tail]).mean(), "prompt not above late"
