"""Double pendule : moteur physique, animation, figures et analyse du chaos."""

from .physics import (
    INTEGRATORS,
    Params,
    Trajectory,
    derivatives,
    from_angular,
    integrate,
    positions,
    rk4,
    to_angular,
    total_energy,
    verlet,
)

__version__ = "0.1.0"
__all__ = [
    "INTEGRATORS",
    "Params",
    "Trajectory",
    "derivatives",
    "from_angular",
    "integrate",
    "positions",
    "rk4",
    "to_angular",
    "total_energy",
    "verlet",
]
