"""Moteur physique du double pendule : Hamiltonien, intégration symplectique, énergie.

On travaille dans l'espace de phase canonique `[θ1, θ2, p1, p2]`, où `p1, p2`
sont les moments conjugués du lagrangien. Cette formulation permet un
intégrateur de Verlet (leapfrog) **symplectique**, qui conserve bien
l'énergie, contrairement à l'intégration naïve en `(θ, θ')`.

Notations :
  Δ = θ2 - θ1
  a  = (m1 + m2) l1²
  b  = m2 l1 l2 cos(Δ)
  c  = m2 l2²
  det = a c - b²
  M^-1 = (1/det) [ [c, -b], [-b, a] ]
  vitesses : [θ1', θ2'] = M^-1 [p1, p2]
  T = ½ [p1, p2] · M^-1 · [p1, p2]ᵀ
  V = -(m1+m2) g l1 cos(θ1) - m2 g l2 cos(θ2)
  H = T + V
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# État = [theta1, theta2, p1, p2]
State = np.ndarray


@dataclass(frozen=True)
class Params:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81


def from_angular(th1: float, th2: float, w1: float, w2: float, p: Params) -> State:
    """Convertit (angles, vitesses angulaires) en coordonnées canoniques."""
    d = th2 - th1
    a = (p.m1 + p.m2) * p.l1 * p.l1
    b = p.m2 * p.l1 * p.l2 * np.cos(d)
    c = p.m2 * p.l2 * p.l2
    p1 = a * w1 + b * w2
    p2 = b * w1 + c * w2
    return np.array([th1, th2, p1, p2])


def to_angular(state: State, p: Params) -> tuple[float, float, float, float]:
    """Retourne (θ1, θ2, ω1, ω2) depuis l'état canonique."""
    th1, th2, p1, p2 = state
    d = th2 - th1
    a = (p.m1 + p.m2) * p.l1 * p.l1
    b = p.m2 * p.l1 * p.l2 * np.cos(d)
    c = p.m2 * p.l2 * p.l2
    det = a * c - b * b
    w1 = (c * p1 - b * p2) / det
    w2 = (-b * p1 + a * p2) / det
    return th1, th2, w1, w2


def _mass_and_forces(state: State, p: Params):
    """Retourne (b, det, ∂T/∂b, ∂V/∂θ1, ∂V/∂θ2)."""
    th1, th2, p1, p2 = state
    d = th2 - th1
    a = (p.m1 + p.m2) * p.l1 * p.l1
    b = p.m2 * p.l1 * p.l2 * np.cos(d)
    c = p.m2 * p.l2 * p.l2
    det = a * c - b * b

    n = c * p1 * p1 - 2.0 * b * p1 * p2 + a * p2 * p2
    db2 = -2.0 * p1 * p2 / det + n * b / (det * det)  # ∂T/∂b
    dv1 = (p.m1 + p.m2) * p.g * p.l1 * np.sin(th1)
    dv2 = p.m2 * p.g * p.l2 * np.sin(th2)
    return b, det, db2, dv1, dv2


def velocities(state: State, p: Params) -> tuple[float, float]:
    """Vitesses angulaires (ω1, ω2) pour un état canonique."""
    th1, th2, p1, p2 = state
    a = (p.m1 + p.m2) * p.l1 * p.l1
    b = p.m2 * p.l1 * p.l2 * np.cos(th2 - th1)
    c = p.m2 * p.l2 * p.l2
    det = a * c - b * b
    return (c * p1 - b * p2) / det, (-b * p1 + a * p2) / det


def total_energy(state: State, p: Params) -> float:
    """Énergie mécanique totale H = T + V (invariant du mouvement)."""
    th1, th2, p1, p2 = state
    d = th2 - th1
    a = (p.m1 + p.m2) * p.l1 * p.l1
    b = p.m2 * p.l1 * p.l2 * np.cos(d)
    c = p.m2 * p.l2 * p.l2
    det = a * c - b * b
    kinetic = 0.5 * (c * p1 * p1 - 2.0 * b * p1 * p2 + a * p2 * p2) / det
    potential = -(p.m1 + p.m2) * p.g * p.l1 * np.cos(th1) - p.m2 * p.g * p.l2 * np.cos(th2)
    return kinetic + potential


def derivatives(state: State, p: Params) -> State:
    """Équations canoniques de Hamilton : [θ1', θ2', p1', p2']."""
    th1, th2, p1, p2 = state
    a = (p.m1 + p.m2) * p.l1 * p.l1
    _, det, db2, dv1, dv2 = _mass_and_forces(state, p)
    b = p.m2 * p.l1 * p.l2 * np.cos(th2 - th1)
    c = p.m2 * p.l2 * p.l2

    # dθ/dt = ∂H/∂p = M^-1 · p
    v1 = (c * p1 - b * p2) / det
    v2 = (-b * p1 + a * p2) / det

    # db/dθ1 = +m2 l1 l2 sin Δ ; db/dθ2 = -m2 l1 l2 sin Δ
    s = p.m2 * p.l1 * p.l2 * np.sin(th2 - th1)
    dp1 = -(db2 * s) - dv1
    dp2 = -(db2 * (-s)) - dv2

    return np.array([v1, v2, dp1, dp2])


def positions(state: State, p: Params) -> tuple[float, float, float, float]:
    """Coordonnées cartésiennes (x1, y1) et (x2, y2) des deux masses.

    y pointe vers le bas (repère écran classique).
    """
    th1, th2 = state[0], state[1]
    x1 = p.l1 * np.sin(th1)
    y1 = p.l1 * np.cos(th1)
    x2 = x1 + p.l2 * np.sin(th2)
    y2 = y1 + p.l2 * np.cos(th2)
    return x1, y1, x2, y2


Integrator = Callable[[float, State, Params], State]


def rk4(dt: float, state: State, p: Params) -> State:
    """Runge-Kutta d'ordre 4 (précis à court terme, non symplectique)."""
    k1 = derivatives(state, p)
    k2 = derivatives(state + 0.5 * dt * k1, p)
    k3 = derivatives(state + 0.5 * dt * k2, p)
    k4 = derivatives(state + dt * k3, p)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def verlet(dt: float, state: State, p: Params) -> State:
    """Leapfrog symplectique (kick-drift-kick) dans l'espace canonique.

    Conserve l'énergie beaucoup mieux que l'Euler / la version (θ, θ').
    """
    # Kick (demi-pas) : n'actualise QUE les moments.
    new = state.copy()
    new[2:] += 0.5 * dt * derivatives(state, p)[2:]

    # Drift (pas complet) sur les angles avec les vitesses mid-pas.
    v1, v2 = velocities(new, p)
    new[0] += dt * v1
    new[1] += dt * v2

    # Rekick (demi-pas) avec les forces au nouvel angle.
    new[2:] += 0.5 * dt * derivatives(new, p)[2:]
    return new


INTEGRATORS: dict[str, Integrator] = {
    "rk4": rk4,
    "verlet": verlet,
}


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    states: State  # shape (n, 4)
    energy: np.ndarray
    params: Params


def integrate(
    initial: State,
    params: Params,
    *,
    dt: float = 1e-3,
    duration: float = 10.0,
    integrator: str = "verlet",
) -> Trajectory:
    """Intègre le double pendule sur `duration` secondes."""
    if integrator not in INTEGRATORS:
        raise ValueError(f"intégrateur inconnu : {integrator!r} (choix: {sorted(INTEGRATORS)})")
    step = INTEGRATORS[integrator]
    n = int(round(duration / dt)) + 1
    times = np.linspace(0.0, duration, n)
    states = np.empty((n, 4))
    states[0] = initial
    energy = np.empty(n)
    energy[0] = total_energy(initial, params)
    s = initial.copy()
    for i in range(1, n):
        s = step(dt, s, params)
        states[i] = s
        energy[i] = total_energy(s, params)
    return Trajectory(times=times, states=states, energy=energy, params=params)
