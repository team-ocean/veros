import os
import sys
import filecmp
import fnmatch
import pkg_resources
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


@pytest.mark.parametrize("setup", SETUPS)
def test_veros_copy_setup(setup, runner, tmpdir):
    result = runner.invoke(veros.cli.veros_copy_setup.cli, [setup, "--to", os.path.join(tmpdir, setup)])
    assert result.exit_code == 0, setup
    assert not result.output

    outpath = os.path.join(tmpdir, setup)
    srcpath = pkg_resources.resource_filename("veros", f"setups/{setup}")
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


def test_import_isolation(tmpdir):
    TEST_KERNEL = dedent(
        """
    import sys
    import veros.cli

    for mod in sys.modules:
        print(mod)
    """
    )

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
