"""Tests de la CLI : conversion des erreurs de validation en `SystemExit` propre.

`run()` délègue toute sa logique à `_run_inner()` sous un unique `try/except
ValueError`, précisément pour que le chemin `--chaos` (qui appelle
`integrate` en interne, via `chaos.max_lyapunov_estimate`/
`chaos.predictability_time`, *avant* d'atteindre l'appel `integrate` "normal"
plus bas dans `_run_inner`) soit couvert au même titre que le chemin par
défaut. `test_invalid_mass_is_caught_even_on_the_chaos_path` vérifie
spécifiquement ce chemin, pour ne pas régresser vers une version qui ne
protégerait que l'appel `integrate` principal.
"""

import pytest

from double_pendulum.cli import build_parser, run


def _parse(argv):
    return build_parser().parse_args(argv)


def test_invalid_length_produces_clean_system_exit(tmp_path):
    args = _parse(["--l1", "0", "--figures", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        run(args)
    assert "l1" in str(exc_info.value)


def test_invalid_dt_produces_clean_system_exit(tmp_path):
    args = _parse(["--dt", "0", "--figures", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        run(args)
    assert "dt" in str(exc_info.value)


def test_invalid_duration_produces_clean_system_exit(tmp_path):
    args = _parse(["--duration", "-1", "--figures", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        run(args)
    assert "duration" in str(exc_info.value)


def test_invalid_mass_is_caught_even_on_the_chaos_path():
    args = _parse(["--m2", "0", "--chaos"])
    with pytest.raises(SystemExit) as exc_info:
        run(args)
    assert "m2" in str(exc_info.value)


def test_valid_params_run_cleanly_via_figures_path(tmp_path):
    args = _parse(["--duration", "0.05", "--dt", "0.01", "--figures", str(tmp_path)])
    result = run(args)
    assert result == 0
    assert (tmp_path / "trajectory.png").exists()
    assert (tmp_path / "phase.png").exists()
    assert (tmp_path / "energy.png").exists()
