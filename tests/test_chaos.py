"""Tests de l'analyse du chaos : divergence et exposant de Lyapunov."""

import numpy as np

from double_pendulum import chaos
from double_pendulum.physics import Params, from_angular


def _chaotic_init() -> tuple[np.ndarray, Params]:
    """Configuration du double pendule en régime fortement chaotique."""
    p = Params()
    return from_angular(np.radians(120), np.radians(90), 3.0, -4.0, p), p


def _regular_init() -> tuple[np.ndarray, Params]:
    """Petite oscillation : mouvement quasi-périodique."""
    p = Params()
    return from_angular(0.15, 0.1, 0.0, 0.0, p), p


def test_perturbed_initial_keeps_others():
    init, _ = _regular_init()
    per = chaos.perturbed_initial(init, 1e-6, axis=0)
    assert per[0] == init[0] + 1e-6
    assert per[1] == init[1]
    assert per[2] == init[2]
    assert per[3] == init[3]


def test_chaotic_initial_diverge():
    init, p = _chaotic_init()
    lyap = chaos.max_lyapunov_estimate(
        init, p, dt=1e-3, duration=20.0, integrator="verlet"
    )
    assert lyap > 0.05


def test_regular_initial_lyapunov_near_zero():
    init, p = _regular_init()
    lyap = chaos.max_lyapunov_estimate(
        init, p, dt=1e-3, duration=20.0, integrator="verlet"
    )
    assert lyap < 0.05


def test_predictability_finite_for_chaos():
    init, p = _chaotic_init()
    pt = chaos.predictability_time(init, p, dt=1e-3, duration=20.0, integrator="verlet")
    assert np.isfinite(pt)
    assert 0.0 < pt <= 20.0


def test_predictability_infinite_for_regular():
    init, p = _regular_init()
    pt = chaos.predictability_time(init, p, dt=1e-3, duration=20.0, integrator="verlet")
    assert pt == float("inf")
