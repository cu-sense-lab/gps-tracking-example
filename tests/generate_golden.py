"""
Capture the golden tracking baseline.

Run BEFORE refactoring, then never again unless a numerical change is intended and
reviewed:

    python -m tests.generate_golden

`tests/test_tracking_regression.py` diffs live output against these files.
"""

from __future__ import annotations

import pathlib

import numpy as np

from .driver import run_scenario
from .scenarios import SCENARIOS

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"


def main() -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        outputs = run_scenario(scenario)
        path = GOLDEN_DIR / f"{scenario.name}.npz"
        np.savez_compressed(path, **outputs)

        # Report the component the loops ran on; for L5 that is the Q pilot, and
        # component 0 would show its data-limited I channel instead.
        component = int(outputs["carrier_component"])
        prompt_mag = np.abs(outputs["prompt_corr"][:, component])
        final_doppler_error = outputs["doppler_freq_hz"][-1] - scenario.doppler_hz
        tail = prompt_mag[-min(500, len(prompt_mag)) :]
        print(
            f"{scenario.name:22} epochs={int(outputs['output_index']):5} "
            f"mode={outputs['final_mode']:3} "
            f"dopp_err={final_doppler_error:+9.3f} Hz "
            f"|P[{component}]|={tail.mean():9.1f} -> {path.name}"
        )


if __name__ == "__main__":
    main()
