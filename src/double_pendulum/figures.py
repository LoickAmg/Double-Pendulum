"""Figures statiques générées à partir d'une trajectoire simulée.

Produit des PNG indépendants de toute fenêtre (backend Agg) :
- trajectoire de la masse 2 dans le plan ;
- portrait de phase (th2 vs th1) et/ou (omega2 vs omega1) ;
- conservation de l'énergie quand on fournit plusieurs intégrateurs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from .animate import _mass2_path
from .physics import Params, Trajectory, integrate


def plot_trajectory(traj: Trajectory, out: str | Path, *, samples: int | None = None) -> Path:
    """Trace le chemin suivi par la masse 2 (kick-starter du chaos visuel)."""
    x, y = _mass2_path(traj)
    if samples is not None:
        step = max(1, len(x) // samples)
        x, y = x[::step], y[::step]

    fig, ax = plt.subplots(figsize=(7, 7))
    p = traj.params
    span = max(p.l1, p.l2) * 2.6
    ax.set_xlim(-span, span)
    ax.set_ylim(span, -span)
    ax.set_aspect("equal")
    ax.plot(x, y, color="#915E4E", lw=0.8, alpha=0.9)
    ax.set_title("Trajectoire de la masse 2 — " + _stamp(traj))
    ax.grid(True, alpha=0.3)
    return _save(fig, out)


def plot_phase_portrait(traj: Trajectory, out: str | Path) -> Path:
    """Portrait de phase dans l'espace (th2, th1)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(np.degrees(traj.states[:, 1]), np.degrees(traj.states[:, 0]), color="#2E6E52", lw=0.5)
    ax.set_xlabel("θ2 (°)")
    ax.set_ylabel("θ1 (°)")
    ax.set_title("Portrait de phase (θ1, θ2)")
    ax.grid(True, alpha=0.3)
    return _save(fig, out)


def plot_energy_compare(
    initial: np.ndarray,
    params: Params,
    *,
    dt: float = 1e-3,
    duration: float = 15.0,
    out: str | Path,
) -> Path:
    """Dérive d'énergie relative pour RK4 vs leapfrog (validation stabilité)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, marker in (("verlet", "o"), ("rk4", "s")):
        traj = integrate(initial, params, dt=dt, duration=duration, integrator=name)
        e0 = traj.energy[0]
        rel = np.abs((traj.energy - e0) / e0)
        ax.plot(traj.times, rel, marker=marker, markevery=50, label=f"{name} (ΔE rel.)")
    ax.set_yscale("log")
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("|ΔE| / E₀")
    ax.set_title("Stabilité : dérive d'énergie par intégrateur")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    return _save(fig, out)


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _stamp(traj: Trajectory) -> str:
    th1, th2 = np.degrees(traj.states[0][:2])
    return f"θ1={th1:.0f}° θ2={th2:.0f}° · É={traj.energy[0]:.3f} J"
