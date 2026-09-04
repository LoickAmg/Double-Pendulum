"""Tests du moteur physique : intégrateurs, énergie, conversions."""

import numpy as np
import pytest

from double_pendulum.physics import (
    Params,
    from_angular,
    integrate,
    positions,
    rk4,
    to_angular,
    total_energy,
    verlet,
)


def _regular() -> tuple[np.ndarray, Params]:
    p = Params()
    return from_angular(0.15, 0.1, 0.0, 0.0, p), p


def test_from_to_angular_roundtrip():
    p = Params()
    s = from_angular(0.8, -1.2, 0.5, -0.7, p)
    th1, th2, w1, w2 = to_angular(s, p)
    assert np.isclose(th1, 0.8)
    assert np.isclose(th2, -1.2)
    assert np.isclose(w1, 0.5)
    assert np.isclose(w2, -0.7)


def test_positions_at_rest():
    p = Params()
    s = from_angular(0.0, 0.0, 0.0, 0.0, p)
    x1, y1, x2, y2 = positions(s, p)
    # Pendu vertical : x = 0, y = l vers le bas.
    assert np.isclose(x1, 0.0) and np.isclose(y1, p.l1)
    assert np.isclose(x2, 0.0) and np.isclose(y2, p.l1 + p.l2)


def test_energy_symmetry_with_angular():
    """Énergie calculée en canonique == celle du point de vue lagrangien."""
    p = Params(m1=1.5, m2=2.0, l1=1.1, l2=0.9, g=9.81)
    th1, th2, w1, w2 = 0.9, -0.6, 1.2, -2.1
    s = from_angular(th1, th2, w1, w2, p)

    d = th2 - th1
    kin = 0.5 * p.m1 * p.l1**2 * w1**2 + 0.5 * p.m2 * (
        p.l1**2 * w1**2 + p.l2**2 * w2**2 + 2 * p.l1 * p.l2 * w1 * w2 * np.cos(d)
    )
    pot = -(p.m1 + p.m2) * p.g * p.l1 * np.cos(th1) - p.m2 * p.g * p.l2 * np.cos(th2)
    assert np.isclose(total_energy(s, p), kin + pot)


def test_verlet_conserves_energy_regular_orbit():
    init, p = _regular()
    traj = integrate(init, p, dt=1e-3, duration=10.0, integrator="verlet")
    rel = np.abs((traj.energy - traj.energy[0]) / traj.energy[0]).max()
    assert rel < 1e-4


def test_rk4_small_drift_regular_orbit():
    init, p = _regular()
    traj = integrate(init, p, dt=1e-3, duration=5.0, integrator="rk4")
    rel = np.abs((traj.energy[-1] - traj.energy[0]) / traj.energy[0])
    assert rel < 1e-3


def test_integrate_unknown_integrator():
    init, p = _regular()
    with pytest.raises(ValueError):
        integrate(init, p, integrator="euler")


def test_single_step_shape_and_finite():
    init, p = _regular()
    for step in (verlet, rk4):
        out = step(0.001, init, p)
        assert out.shape == (4,)
        assert np.all(np.isfinite(out))
