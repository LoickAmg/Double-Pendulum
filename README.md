# Double Pendulum

Simulation du **double pendule** : intégration numérique, animation temps réel,
figures statiques et **analyse du chaos**. Un projet de la
[roadmap 40 projets](../roadmap-40-projets.md) (#6).

## Architecture

```
src/double_pendulum/
  physics.py   Hamiltonien, intégration symplectique (Verlet) & RK4, énergie
  chaos.py     divergence de trajectoires proches, exposant de Lyapunov,
               temps de prévisibilité
  animate.py   animation live Matplotlib (+ export gif/mp4)
  figures.py   figures statiques (trajectoire, portrait de phase, énergie)
  cli.py       point d'entrée en ligne de commande
tests/
  test_physics.py    intégrateurs, énergie, conversions canoniques
  test_chaos.py      chaos vs régulier, prévisibilité
```

## Physique

- Coordonnées canoniques `[θ1, θ2, p1, p2]` (moments conjugués du lagrangien).
- Équations de Hamilton résolues **analytiquement** (inverse de la matrice de
  masse) — vérifiées contre un gradient numérique.
- **Verlet (leapfrog symplectique) kick-drift-kick** : conserve l'énergie à
  ~1e-6 sur les orbites régulières ; **RK4** précis à court terme mais non
  symplectique.
- Dans le régime fortement chaotique (retournements au-dessus du point
  d'appui), même un intégrateur symplectique dérive : c'est la signature du
  chaos, mesurée par l'exposant de Lyapunov.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
.venv\Scripts\pip install -e ".[test]"
```

## Usage

```bash
# Animation interactive (fenêtre)
double-pendulum --angles 120 90 --duration 15

# Export vidéo / gif (backend Agg, sans fenêtre)
double-pendulum --save out.gif --angles 120 90 --duration 5
double-pendulum --save out.mp4 --duration 8

# Figures statiques (trajectoire, portrait de phase, stabilité énergie)
double-pendulum --figures figures --duration 10

# Analyse du chaos : Lyapunov + temps de prévisibilité
double-pendulum --chaos --duration 30
```

Options utiles : `--m1 --m2 --l1 --l2 --g --dt --integrator rk4|verlet`.

## Exemple de sortie `--chaos`

```
Double pendule (theta1=120.0 deg, theta2=90.0 deg)
  Exposant de Lyapunov estime : +1.420 /s
  Temps de previsibilite (epsilon=1e-6) : 11.82 s
  -> mouvement chaotique (sensibilite aux conditions initiales)
```

À petite amplitude, l'exposant retombe vers 0 : mouvement quasi-périodique
(prévisible), signe de la transition régulier → chaotique.

## Tests & qualité

```bash
pytest                     # 12 tests
ruff check src tests       # propre
```

## Projections

- Diagramme de bifurcation en fonction de l'énergie / des amplitudes initiales.
- Cartes de Lyapunov sur une grille de conditions initiales.
- Export du portrait de phase stroboscopique (section de Poincaré).

## Licence

MIT — voir [LICENSE](LICENSE).
