"""Point d'entrée en ligne de commande du double pendule."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import animate, chaos, figures
from .physics import Params, from_angular, integrate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="double-pendulum",
        description="Double pendule : animation, figures statiques et analyse du chaos.",
    )
    parser.add_argument("--angles", nargs=2, type=float, default=[120.0, 90.0],
                        metavar=("DEG1", "DEG2"),
                        help="angles initiaux θ1, θ2 en degrés (défaut: 120 90)")
    parser.add_argument("--m1", type=float, default=1.0, help="masse 1 (défaut: 1)")
    parser.add_argument("--m2", type=float, default=1.0, help="masse 2 (défaut: 1)")
    parser.add_argument("--l1", type=float, default=1.0, help="longueur 1 (défaut: 1)")
    parser.add_argument("--l2", type=float, default=1.0, help="longueur 2 (défaut: 1)")
    parser.add_argument("--g", type=float, default=9.81, help="gravité (défaut: 9.81)")
    parser.add_argument("--integrator", choices=["rk4", "verlet"], default="verlet",
                        help="schéma d'intégration (défaut: verlet)")
    parser.add_argument("--dt", type=float, default=1e-3, help="pas de temps (défaut: 0.001)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="durée en secondes (défaut: 15)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", metavar="FILE", default=None,
                       help="écrire une animation (img.gif ou img.mp4) au lieu d'ouvrir la fenêtre")
    group.add_argument("--figures", metavar="DIR", default=None,
                       help="générer les figures statiques dans DIR")
    group.add_argument("--chaos", action="store_true",
                       help="analyser le chaos (Lyapunov, prévisibilité)")

    parser.add_argument("--trail", type=int, default=120,
                        help="longueur de la trace (défaut: 120)")
    parser.add_argument("--interval", type=int, default=20,
                        help="intervalle d'animation en ms (défaut: 20)")
    return parser


def run(args: argparse.Namespace) -> int:
    # `Params.__post_init__` (masses/longueurs) et `integrate` (dt/duration)
    # valident déjà ces paramètres physiques (voir physics.py) et lèvent
    # `ValueError` — toute la logique de la commande est donc regroupée
    # dans `_run_inner` et appelée ici sous un seul `try` pour convertir
    # n'importe laquelle de ces erreurs (y compris celles du chemin
    # `--chaos`, qui appelle `integrate` avant d'atteindre le `traj = ...`
    # plus bas) en erreur CLI propre plutôt qu'en trace Python.
    try:
        return _run_inner(args)
    except ValueError as exc:
        raise SystemExit(f"double-pendulum: paramètre invalide : {exc}") from exc


def _run_inner(args: argparse.Namespace) -> int:
    params = Params(m1=args.m1, m2=args.m2, l1=args.l1, l2=args.l2, g=args.g)
    initial = from_angular(
        np.radians(args.angles[0]), np.radians(args.angles[1]), 0.0, 0.0, params
    )

    if args.chaos:
        lyap = chaos.max_lyapunov_estimate(
            initial, params, dt=args.dt, duration=max(args.duration, 30.0),
            integrator=args.integrator,
        )
        pred = chaos.predictability_time(
            initial, params, dt=args.dt, duration=max(args.duration, 30.0),
            integrator=args.integrator,
        )
        print(f"Double pendule (theta1={args.angles[0]:.1f} deg, theta2={args.angles[1]:.1f} deg)")
        print(f"  Exposant de Lyapunov estime : {lyap:+.3f} /s")
        print(f"  Temps de previsibilite (epsilon=1e-6) : {_fmt_time(pred)}")
        if lyap > 0.05:
            print("  -> mouvement chaotique (sensibilite aux conditions initiales)")
        else:
            print("  -> mouvement quasi-periodique (Lyapunov proche de 0)")
        return 0

    traj = integrate(initial, params, dt=args.dt, duration=args.duration,
                     integrator=args.integrator)

    if args.figures:
        from pathlib import Path

        outdir = Path(args.figures)
        figures.plot_trajectory(traj, outdir / "trajectory.png")
        figures.plot_phase_portrait(traj, outdir / "phase.png")
        figures.plot_energy_compare(initial, params, dt=args.dt, duration=args.duration,
                                    out=outdir / "energy.png")
        print(f"Figures ecrites dans {outdir.resolve()}")
        return 0

    from matplotlib import pyplot as plt

    animate.animate(traj, save=args.save, interval_ms=args.interval, trail=args.trail)
    if args.save:
        print(f"Animation ecrite dans {args.save}")
    else:
        plt.show()  # fenêtre interactive (bloquant)
    return 0


def _fmt_time(t: float) -> str:
    if t == float("inf"):
        return "infini (aucune divergence observee)"
    return f"{t:.2f} s"


def main(argv: Sequence[str] | None = None) -> int:
    # Consoles Windows (cp1252) : passe en UTF-8 pour afficher θ, °, …
    # sans planter sur les caractères non-latins.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
