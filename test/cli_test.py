import os
import sys
import filecmp
import fnmatch
import subprocess
from textwrap import dedent

from click.testing import CliRunner
import pytest

import veros.cli

SETUPS = (
    "acc",
    "acc_basic",
    "global_4deg",
    "global_1deg",
    "global_flexible",
    "north_atlantic",
)


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.fixture
def runtime_options():
    from veros import runtime_settings as rs

    original = {name: getattr(rs, name) for name in rs.__settings__}
    original["__locked__"] = rs.__locked__
    try:
        yield rs
    finally:
        for name, value in original.items():
            object.__setattr__(rs, name, value)


@pytest.mark.parametrize("setup", SETUPS)
def test_veros_copy_setup(setup, runner, tmpdir):
    import veros

    outpath = os.path.join(tmpdir, setup)
    result = runner.invoke(veros.cli.veros_copy_setup.cli, [setup, "--to", outpath])
    assert result.exit_code == 0, setup
    assert not result.output

    srcpath = os.path.join(os.path.dirname(veros.__file__), "setups", setup)
    ignore = [
        f
        for f in os.listdir(srcpath)
        if any(fnmatch.fnmatch(f, pattern) for pattern in veros.cli.veros_copy_setup.IGNORE_PATTERNS)
    ]

    comparer = filecmp.dircmp(outpath, srcpath, ignore=ignore)
    assert not comparer.left_only and not comparer.right_only

    with open(os.path.join(outpath, f"{setup}.py"), "r") as f:
        setup_content = f.read()

    assert "VEROS_VERSION" in setup_content


def test_veros_run(runner, tmpdir, monkeypatch, runtime_options):
    from veros import VerosSetup

    backend = runtime_options.backend

    setup = "acc"
    runs = []
    run = VerosSetup.run

    def capture_run(simulation, **kwargs):
        result = run(simulation, **kwargs)
        runs.append((simulation.execution_mode, int(simulation.state.variables.itt)))
        return result

    monkeypatch.setattr(VerosSetup, "run", capture_run)

    with runner.isolated_filesystem(tmpdir):
        result = runner.invoke(veros.cli.veros_copy_setup.cli, [setup])

        args = [os.path.join(setup, f"{setup}.py"), "--backend", backend, "--diskless-mode"]
        for name, value in dict(nx=8, ny=8, dt_mom=600, dt_tracer=600, runlen=1200).items():
            args.extend(("--override", name, str(value)))

        object.__setattr__(runtime_options, "__locked__", False)
        result = runner.invoke(veros.cli.veros_run.cli, args)

        assert result.exit_code == 0, result.output
        assert runs == [("compiled" if backend == "jax" else "eager", 2)]


def test_import_isolation(tmpdir):
    TEST_KERNEL = dedent("""
    import sys
    import veros.cli

    for mod in sys.modules:
        print(mod)
    """)

    tmpfile = tmpdir / "isolation.py"
    with open(tmpfile, "w") as f:
        f.write(TEST_KERNEL)

    proc = subprocess.run([sys.executable, tmpfile], check=True, capture_output=True, text=True)

    imported_modules = proc.stdout.split()
    veros_modules = [mod for mod in imported_modules if mod.startswith("veros.")]

    for mod in veros_modules:
        assert mod.startswith("veros.cli") or mod == "veros._version"

    # make sure using the CLI does not initialize MPI
    assert "mpi4py" not in imported_modules


def test_compiled_requires_jax(runner, tmp_path):
    setup = tmp_path / "setup.py"
    setup.write_text("")
    result = runner.invoke(veros.cli.veros_run.cli, [str(setup), "--compiled"])
    assert result.exit_code == 2
    assert "--compiled requires --backend jax" in result.output


def test_compiled_solver_requires_compiled(runner, tmp_path):
    setup = tmp_path / "setup.py"
    setup.write_text("")
    result = runner.invoke(veros.cli.veros_run.cli, [str(setup), "--compiled-solver", "petsc"])
    assert result.exit_code == 2
    assert "--compiled-solver requires --compiled" in result.output


@pytest.mark.parametrize(
    "options, execution, solver",
    [
        ([], "auto", None),
        (["--execution", "eager"], "eager", None),
        (["--execution", "compiled"], "compiled", None),
        (["--compiled"], "compiled", None),
        (["--execution", "compiled", "--compiled-solver", "scipy_jax"], "compiled", "scipy_jax"),
    ],
)
def test_execution_options_route_to_runner(runner, tmp_path, monkeypatch, runtime_options, options, execution, solver):
    from types import SimpleNamespace
    from veros.setups.acc_basic import ACCBasicSetup

    calls = []

    class Setup(ACCBasicSetup):
        def __init__(self, **kwargs):
            pass

        def setup(self):
            calls.append("setup")

        def run(self, **kwargs):
            calls.append(kwargs)

    setup = tmp_path / "setup.py"
    setup.write_text("")
    monkeypatch.setattr(veros.cli.veros_run, "_import_from_file", lambda path: SimpleNamespace(Setup=Setup))
    object.__setattr__(runtime_options, "__locked__", False)
    result = runner.invoke(veros.cli.veros_run.cli, [str(setup), "--backend", "jax", *options])
    assert result.exit_code == 0, result.output
    expected = {}
    if execution != "auto":
        expected["execution"] = execution
    if solver is not None:
        expected["linear_solver"] = solver
    assert calls == ["setup", expected]


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.parametrize("accepts_progress", (False, True))
@pytest.mark.parametrize("options", ([], ["--execution", "compiled"]))
def test_legacy_run_signature(runner, tmp_path, monkeypatch, runtime_options, backend, accepts_progress, options):
    from types import SimpleNamespace
    from veros.setups.acc_basic import ACCBasicSetup

    calls = []

    class Setup(ACCBasicSetup):
        def __init__(self, **kwargs):
            pass

        def setup(self):
            calls.append("setup")

    if accepts_progress:

        def run(self, show_progress_bar=None):
            calls.append("run")

    else:

        def run(self):
            calls.append("run")

    Setup.run = run
    setup = tmp_path / "setup.py"
    setup.write_text("")
    monkeypatch.setattr(veros.cli.veros_run, "_import_from_file", lambda path: SimpleNamespace(Setup=Setup))
    object.__setattr__(runtime_options, "__locked__", False)
    result = runner.invoke(veros.cli.veros_run.cli, [str(setup), "--backend", backend, *options])
    if options:
        assert result.exit_code == 2, result.output
        assert "run" not in calls
        if backend == "jax":
            assert "run() does not accept" in result.output
    else:
        assert result.exit_code == 0, result.output
        assert calls == ["setup", "run"]


def test_run_override_accepting_execution_only(runner, tmp_path, monkeypatch, runtime_options):
    from types import SimpleNamespace
    from veros.setups.acc_basic import ACCBasicSetup

    calls = []

    class Setup(ACCBasicSetup):
        def __init__(self, **kwargs):
            pass

        def setup(self):
            calls.append("setup")

        def run(self, *, execution="auto"):
            calls.append(execution)

    setup = tmp_path / "setup.py"
    setup.write_text("")
    monkeypatch.setattr(veros.cli.veros_run, "_import_from_file", lambda path: SimpleNamespace(Setup=Setup))
    object.__setattr__(runtime_options, "__locked__", False)
    result = runner.invoke(veros.cli.veros_run.cli, [str(setup), "--backend", "jax", "--execution", "compiled"])
    assert result.exit_code == 0, result.output
    assert calls == ["setup", "compiled"]


def test_compiled_alias_conflicts_with_eager(runner, tmp_path):
    setup = tmp_path / "setup.py"
    setup.write_text("")
    result = runner.invoke(veros.cli.veros_run.cli, [str(setup), "--compiled", "--execution", "eager"])
    assert result.exit_code == 2
    assert "conflicts" in result.output
