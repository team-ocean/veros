"""Check report integrity, acceptance isolation, and column physics."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.ad_demos.__main__ import main, write_result
from benchmarks.ad_demos.common import Checks


def test_nonfinite_values_fail_and_serialize(tmp_path):
    checks = Checks()
    checks.close("nonfinite", np.nan, 1)
    assert not checks.items[0]["passed"]
    destination = tmp_path / "result.json"
    write_result(destination, dict(checks=checks.items, metric=np.array([1, np.inf])))
    result = json.loads(destination.read_text())
    assert result["checks"][0]["detail"]["actual"] is None
    assert result["metric"] == [1, None]


@pytest.mark.parametrize("status,exit_code", [("smoke_passed", 0), ("smoke_failed", 1), ("error", 1)])
def test_parent_preserves_child_failure(monkeypatch, tmp_path, status, exit_code):
    def child(command, **kwargs):
        destination = command[command.index("--output") + 1]
        write_result(tmp_path / "closure_learning.json", dict(demo="closure_learning", status=status))
        assert destination == str(tmp_path / "closure_learning.json")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", child)
    monkeypatch.setattr(sys, "argv", ["demo", "--demo", "closure_learning", "--output", str(tmp_path)])
    assert main() == exit_code
    assert json.loads((tmp_path / "summary.json").read_text())["results"][0]["status"] == status


def test_timeout_cannot_reuse_stale_pass(monkeypatch, tmp_path):
    destination = tmp_path / "closure_learning.json"
    write_result(destination, dict(status="passed"))

    def timeout(*args, **kwargs):
        assert not destination.exists()
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    monkeypatch.setattr(sys, "argv", ["demo", "--demo", "closure_learning", "--output", str(tmp_path)])
    assert main() == 1
    assert json.loads(destination.read_text())["status"] == "timeout"


def test_timeout_retains_completed_checks(monkeypatch, tmp_path):
    def timeout(command, **kwargs):
        write_result(
            tmp_path / "closure_learning.json",
            dict(status="running", checks=[dict(name="conservation", passed=True)], metrics=dict(error=1e-12)),
        )
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    monkeypatch.setattr(sys, "argv", ["demo", "--demo", "closure_learning", "--output", str(tmp_path)])
    assert main() == 1
    report = json.loads((tmp_path / "closure_learning.json").read_text())
    assert report["status"] == "timeout"
    assert report["checks"][0]["passed"]
    assert report["metrics"]["error"] == 1e-12


def test_gradient_harness_accepts_affine_and_quadratic():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import gradient_checks

    jax.config.update("jax_enable_x64", True)
    for objective in (lambda z: jnp.sum(z), lambda z: jnp.sum(z**2)):
        checks = Checks()
        gradient_checks(objective, jnp.array([0.3, 0.2]), checks)
        assert all(item["passed"] for item in checks.items), checks.items


def test_gradient_harness_rejects_consistent_but_wrong_ad():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import gradient_checks

    @jax.custom_jvp
    def objective(z):
        return jnp.sum(z)

    @objective.defjvp
    def wrong_derivative(primals, tangents):
        return objective(primals[0]), 2 * jnp.sum(tangents[0])

    checks = Checks()
    gradient_checks(objective, jnp.array([0.3, 0.2]), checks)
    outcomes = {item["name"]: item["passed"] for item in checks.items}
    assert outcomes["forward/reverse directional derivative"]
    assert not outcomes["finite-difference plateau"]
    assert not outcomes["Taylor remainder"]


def test_column_conservation_and_equilibrium():
    # Configure MPI before importing Veros in this serial-only test.
    if "veros" not in sys.modules:
        try:
            import mpi4py
        except ImportError:
            pass
        else:
            mpi4py.rc.initialize = False
    from veros import runtime_settings

    if "veros.core.operators" not in sys.modules:
        runtime_settings.mpi_comm = None
    if runtime_settings.backend != "jax":
        pytest.skip("column TDMA test requires --backend jax")
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.column import Column

    jax.config.update("jax_enable_x64", True)
    column = Column()
    fitted = jax.jit(lambda x: column.rollout(x, jnp.array([0.3, 0.2]), 12))(column.initial)
    np.testing.assert_allclose(jnp.sum(fitted), jnp.sum(column.initial), rtol=1e-12)
    assert float(jnp.min(fitted)) >= float(jnp.min(column.initial)) - 1e-12
    assert float(jnp.max(fitted)) <= float(jnp.max(column.initial)) + 1e-12

    column = Column(dt=86400, restoring=True)
    a, b, c, _ = column.coefficients(column.initial, jnp.array([0.3, 0]))
    equilibrium = column.solve(a, b, c, column.source())
    next_state = column.step(equilibrium, jnp.array([0.3, 0]))
    np.testing.assert_allclose(next_state, equilibrium, rtol=1e-11)
