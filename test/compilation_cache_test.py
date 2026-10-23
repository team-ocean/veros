"""Exercise cache configuration and executable reuse in fresh processes."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = """
import json
import jax
import jax.numpy as jnp
from veros.core.density import gsw
value = gsw.gsw_rho(jnp.full((4, 5, 6), 35.), 12., 1000.)
jax.block_until_ready(value)
print(json.dumps(dict(cache=jax.config.jax_compilation_cache_dir, value=value.tolist())))
"""


def run_kernel(environment):
    return subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="requires JAX")
@pytest.mark.parametrize("veros_cache", (False, True))
def test_cache_configuration_and_reuse(tmp_path, veros_cache):
    environment = dict(os.environ, VEROS_BACKEND="jax")
    for key in ("VEROS_JAX_COMPILATION_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR"):
        environment.pop(key, None)
    key = "VEROS_JAX_COMPILATION_CACHE_DIR" if veros_cache else "JAX_COMPILATION_CACHE_DIR"
    environment.update(
        {
            key: str(tmp_path),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
            "JAX_LOG_COMPILES": "true",
            "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
        }
    )
    if veros_cache:
        environment["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "other")
    cold = run_kernel(environment)
    warm = run_kernel(environment)
    cold_value = json.loads(cold.stdout.splitlines()[-1])
    assert cold_value["cache"] == str(tmp_path)
    assert cold_value == json.loads(warm.stdout.splitlines()[-1])
    assert "Persistent compilation cache hit for 'jit_gsw_rho'" in warm.stderr
    assert any(tmp_path.iterdir())


def test_cache_setting_unused_by_numpy_backend(tmp_path):
    environment = dict(os.environ, VEROS_BACKEND="numpy", VEROS_JAX_COMPILATION_CACHE_DIR=str(tmp_path / "unused"))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from veros.core.density import gsw; "
            "gsw.gsw_rho(35., 12., 1000.); assert 'jax' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        timeout=60,
    )
    assert not (tmp_path / "unused").exists()
