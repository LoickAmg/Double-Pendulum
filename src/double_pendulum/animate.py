"""Animation temps réel du double pendule avec Matplotlib.

Affiche une fenêtre interactive ou, en mode non interactif (`--save`),
écrit une vidéo/gif à partir de la trajectoire calculée.

Le pas d'intégration (`dt`) est généralement bien plus fin que la fréquence
d'affichage : on sous-échantillonne les états pour que `interval_ms` corresponde
à peu près au temps réel.
"""

from __future__ import annotations

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from .physics import Trajectory, positions

# Nombre maximal d'images pour un fichier enregistré (évite les fichiers énormes).
_MAX_SAVE_FRAMES = 480


def _display_indices(traj: Trajectory, interval_ms: int, save: bool) -> np.ndarray:
    """Indices des états à afficher, sous-échantillonnés pour un rendu fluide."""
    n = len(traj.times)
    if n <= 1:
        return np.array([0])
    if save:
        # Images régulièrement espacées, plafonnées.
        step = max(1, n // _MAX_SAVE_FRAMES)
        return np.arange(0, n, step, dtype=int)
    # Approx temps réel : chaque image avance le temps de `interval_ms`.
    dt = traj.times[1] - traj.times[0]
    real_step = max(1, int(round((interval_ms / 1000.0) / dt)))
    return np.arange(0, n, real_step, dtype=int)


def animate(
    traj: Trajectory,
    *,
    save: str | None = None,
    interval_ms: int = 20,
    dpi: int = 110,
    trail: int = 120,
) -> animation.FuncAnimation:
    """Anime le pendule le long de `traj`.

    - `save=None` : ouvre la fenêtre interactive.
    - `save='out.mp4'|'out.gif'` : écrit un fichier (backend Agg).
    """
    p = traj.params
    x2, y2 = _mass2_path(traj)
    idx = _display_indices(traj, interval_ms, bool(save))
    # Taille de la fenêtre de trace, exprimée en indices d'état.
    stride = (idx[1] - idx[0]) if len(idx) > 1 else 1
    trail_state = max(1, stride * trail)

    fig, ax = plt.subplots(figsize=(7, 7))
    span = max(p.l1, p.l2) * 2.6
    ax.set_xlim(-span, span)
    ax.set_ylim(span, -span)  # y vers le bas
    ax.set_aspect("equal")
    ax.set_title("Double pendule — " + _stamp(traj))
    ax.grid(True, alpha=0.3)

    (rod1,) = ax.plot([], [], "-o", color="#2E6E52", lw=3, ms=8, zorder=3)
    (rod2,) = ax.plot([], [], "-o", color="#B07030", lw=3, ms=8, zorder=3)
    (trace,) = ax.plot([], [], color="#915E4E", lw=1, alpha=0.7, zorder=1)
    text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9)

    def frame(i: int):
        x1, y1, xx2, yy2 = positions(traj.states[i], p)
        rod1.set_data([0, x1], [0, y1])
        rod2.set_data([x1, xx2], [y1, yy2])
        lo = max(0, i - trail_state)
        trace.set_data(x2[lo : i + 1], y2[lo : i + 1])
        text.set_text(f"t = {traj.times[i]:.2f} s   E = {traj.energy[i]:.3f} J")
        return rod1, rod2, trace, text

    anim = animation.FuncAnimation(
        fig, frame, frames=idx, interval=interval_ms, blit=True
    )

    if save:
        writer = "pillow" if save.lower().endswith(".gif") else "ffmpeg"
        anim.save(save, writer=writer, dpi=dpi, fps=1000 // max(interval_ms, 1))
        plt.close(fig)
    return anim


def _stamp(traj: Trajectory) -> str:
    th1, th2 = np.degrees(traj.states[0][:2])
    return f"theta1={th1:.0f} deg, theta2={th2:.0f} deg · E={traj.energy[0]:.3f} J"


def _mass2_path(traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array([positions(s, traj.params) for s in traj.states])
    return coords[:, 2], coords[:, 3]
