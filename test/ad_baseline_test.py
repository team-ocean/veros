"""Profiling reports preserve numerical failures and resource stops."""

import json
import sys

import numpy as np
import pytest

from benchmarks.ad_demos.baseline import guarded
from benchmarks.ad_demos.common import descend
from benchmarks.ad_demos.provenance import collect, file_hash

if "veros" not in sys.modules:
    try:
        import mpi4py
    except ImportError:
        pass
    else:
        mpi4py.rc.initialize = False


def test_optimizer_diagnostics_preserve_the_update_and_detect_stopping():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    point = jnp.zeros(2)

    def objective(z):
        return jnp.sum((z - jnp.array([0.3, 0.4])) ** 2)

    original, original_history = descend(objective, point, iterations=24, radius=1)
    diagnostics = {}
    measured, measured_history = descend(objective, point, iterations=24, radius=1, diagnostics=diagnostics)
    np.testing.assert_array_equal(original, measured)
    assert original_history == measured_history
    assert diagnostics["stopping"] == "gradient_tolerance"
    assert diagnostics["projected_gradient_norm"] < 1e-12
    assert measured_history[-1] < 1e-24
    diagnostics = {}
    descend(lambda z: jnp.sum((z - 0.0002) ** 2), point, diagnostics=diagnostics)
    assert diagnostics["stopping"] == "backtracking_exhausted"
    assert diagnostics["gradient_norm"] > diagnostics["gradient_tolerance"]


def test_guard_preserves_incremental_failures_on_timeout(tmp_path):
    pytest.importorskip("psutil")
    destination = tmp_path / "partial.json"
    program = (
        "import json, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'running','checks':[{'passed':False}]})); "
        "time.sleep(30)"
    )
    result = guarded(
        [sys.executable, "-c", program, str(destination)], destination, timeout=1, memory_mib=1000, reserve_mib=0
    )
    assert result["status"] == "resource_stop"
    assert result["stop_reason"] == "timeout"
    assert result["checks"] == [dict(passed=False)]
    assert result["resource_budget"]["sampled_peak_process_tree_mib"] > 0


def test_guard_cannot_reuse_stale_success(tmp_path):
    pytest.importorskip("psutil")
    destination = tmp_path / "stale.json"
    destination.write_text(json.dumps(dict(status="measured", checks=[dict(passed=True)])))
    result = guarded(
        [sys.executable, "-c", "raise SystemExit(1)"], destination, timeout=5, memory_mib=1000, reserve_mib=0
    )
    assert result["status"] == "error"
    assert "checks" not in result


def test_asset_fingerprint_changes_with_contents(tmp_path):
    asset = tmp_path / "forcing.nc"
    asset.write_bytes(b"first")
    before = file_hash(asset)
    asset.write_bytes(b"second")
    assert file_hash(asset) != before


def test_source_snapshot_matches_reported_fingerprints(tmp_path):
    import hashlib
    import zipfile

    provenance = collect(dict(setup="acc"), source_directory=tmp_path)
    with zipfile.ZipFile(provenance["source_snapshot"]) as archive:
        assert set(archive.namelist()) == set(provenance["source_sha256"])
        for name in archive.namelist():
            assert hashlib.sha256(archive.read(name)).hexdigest() == provenance["source_sha256"][name]


@pytest.mark.parametrize("products", [("checkpoint", "tangent", "finite_difference"), ("finite_difference",)])
def test_profile_products_on_an_understood_stable_recurrence(monkeypatch, products):
    from collections import namedtuple
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from veros import runtime_settings
    from veros.integration import IntegrationState
    from benchmarks.ad_demos.baseline import profile
    from benchmarks.ad_demos import research_ocean

    if runtime_settings.mpi_comm is not None:
        runtime_settings.mpi_comm = None
    jax.config.update("jax_enable_x64", True)
    Variables = namedtuple("Variables", "x time Nsqr maskW tau")

    class Prepared:
        def __init__(self):
            self._compiled_step = jax.jit(self.numerical_step)

        @staticmethod
        def numerical_step(carry, parameters):
            vs = carry.variables
            return carry._replace(
                variables=vs._replace(x=parameters["gain"] * vs.x + parameters["forcing"], time=vs.time + 1)
            )

        def step(self, carry, *, parameters):
            return self._compiled_step(carry, parameters), {}

        def advance(self, carry, *, n_steps, parameters, **kwargs):
            for _ in range(n_steps):
                carry = self._compiled_step(carry, parameters)
            return carry, {}

        @staticmethod
        def validate_parameters(parameters):
            return parameters

        @staticmethod
        def check_status(carry):
            assert int(carry.failure_code) == 0

    class Ocean:
        dt = 86400.0
        heat_index, amoc_index = 0, 1

        def __init__(self, *args):
            self.initial = IntegrationState(
                Variables(
                    jnp.array(1.0, dtype="float64"),
                    jnp.array(0.0, dtype="float64"),
                    jnp.ones((5, 5, 2, 3)),
                    jnp.ones((5, 5, 2), bool),
                    jnp.array(0, dtype="int32"),
                ),
                {},
                jnp.array(0, dtype="int32"),
                jnp.array(-1, dtype="int32"),
            )
            self.prepared = Prepared()
            self.sim = SimpleNamespace(state=SimpleNamespace(settings=dict(nx=1, ny=1, nz=2)))

        def controls(self, full):
            return self.initial._replace(variables=self.initial.variables._replace(x=1 + full[0])), {
                "gain": 0.8 + 0.01 * full[8],
                "forcing": full[2] + 0.2 * full[4] + 0.5 * full[9],
            }

        @staticmethod
        def observe(carry):
            return jnp.stack((carry.variables.x, carry.variables.x**2))

    monkeypatch.setattr(research_ocean, "ResearchOcean", Ocean)
    report = {}
    config = dict(
        steps=7,
        spinup_steps=0,
        objective="heat",
        products=products,
        control_family="joint",
        neutral_policy="default",
        fd_epsilons=[1e-3, 1e-4],
    )
    profile(config, report, lambda: None)
    assert all(item["passed"] for item in report["checks"]), report["checks"]
    np.testing.assert_allclose(report["metrics"]["forward_value"], 0.8**7, rtol=1e-12)
    if "checkpoint" in products:
        assert report["metrics"]["checkpoint"]["reverse_steps"] == 7
        assert report["metrics"]["checkpoint"]["peak_ram_checkpoints"] <= 4
