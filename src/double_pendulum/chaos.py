"""Analyse du chaos : sensibilité aux conditions initiales.

La marque du chaos déterministe est la forte sensibilité aux conditions
initiales : deux trajectoires qui partent d'états quasi identiques divergent
exponentiellement en moyenne. On mesure l'écart (dans l'espace des angles)
entre une trajectoire de référence et une trajectoire perturbée, et on en
déduit un exposant de Lyapunov estimé ainsi qu'un temps de prévisibilité.
"""

from __future__ import annotations

import numpy as np

from .physics import Params, Trajectory, integrate

# Distance maximale dans l'espace des angles (th1, th2) : saturation.
_ANGLE_RANGE = 2.0 * np.pi


def perturbed_initial(initial: np.ndarray, epsilon: float, axis: int = 0) -> np.ndarray:
    """Copie `initial` avec une petite perturbation `epsilon` sur `axis`."""
    out = initial.copy()
    out[axis] += epsilon
    return out


def divergence(reference: Trajectory, perturbed: Trajectory) -> np.ndarray:
    """Distance euclidienne entre les deux trajectoires dans (th1, th2)."""
    d = reference.states - perturbed.states
    return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)


def max_lyapunov_estimate(
    initial: np.ndarray,
    params: Params,
    *,
    epsilon: float = 1e-6,
    dt: float = 1e-3,
    duration: float = 30.0,
    integrator: str = "verlet",
) -> float:
    """Exposant de Lyapunov maximal estimé par la pente initiale de log(d(t)).

    On intègre la trajectoire de référence et une trajectoire perturbée, on
    mesure l'écart `d`, puis on ajuste une droite sur `log(d)` pendant la
    phase où l'écart reste linéaire (avant saturation à l'échelle du système).
    """
    ref = integrate(initial, params, dt=dt, duration=duration, integrator=integrator)
    pen = integrate(
        perturbed_initial(initial, epsilon, 0),
        params,
        dt=dt,
        duration=duration,
        integrator=integrator,
    )
    dist = divergence(ref, pen)

    # Phase de croissance : tant que l'écart n'a pas saturé, log(d) est ~ linéaire.
    saturation = np.pi / 2.0
    mask = dist <= saturation
    if mask.sum() < 3:
        # Trop peu de points dans la phase linéaire : mouvement régulier.
        return 0.0

    t = ref.times[mask]
    logd = np.log(np.clip(dist[mask], 1e-12, None))
    slope, _ = np.polyfit(t, logd, 1)
    return float(slope)


def predictability_time(
    initial: np.ndarray,
    params: Params,
    *,
    epsilon: float = 1e-6,
    dt: float = 1e-3,
    duration: float = 30.0,
    integrator: str = "verlet",
) -> float:
    """Temps (s) au bout duquel une perturbation `epsilon` devient de l'ordre de
    l'amplitude du mouvement (mesure de prévisibilité)."""
    ref = integrate(initial, params, dt=dt, duration=duration, integrator=integrator)
    pen = integrate(
        perturbed_initial(initial, epsilon, 0),
        params,
        dt=dt,
        duration=duration,
        integrator=integrator,
    )
    dist = divergence(ref, pen)
    threshold = np.pi / 2.0
    exceed = np.flatnonzero(dist >= threshold)
    if exceed.size == 0:
        return float("inf")
    first = exceed[0]
    if first == 0:
        return 0.0
    return float(ref.times[first])
