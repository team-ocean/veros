import os
import filecmp
import fnmatch
import pkg_resources

from click.testing import CliRunner
import pytest

import veros.cli

SETUPS = ('acc', 'acc_basic', 'acc_sector', 'global_4deg', 'global_1deg',
          'global_flexible', 'north_atlantic', 'wave_propagation')


@pytest.fixture(scope='module')
def runner():
    return CliRunner()


@pytest.mark.parametrize("setup", SETUPS)
def test_veros_copy_setup(setup, runner, tmpdir):
    result = runner.invoke(veros.cli.veros_copy_setup.cli, [setup, '--to', os.path.join(tmpdir, setup)])
    assert result.exit_code == 0, setup
    assert not result.output

    outpath = os.path.join(tmpdir, setup)
    srcpath = pkg_resources.resource_filename('veros', f'setups/{setup}')
    ignore = [f for f in os.listdir(srcpath) if any(
        fnmatch.fnmatch(f, pattern) for pattern in veros.cli.veros_copy_setup.IGNORE_PATTERNS
    )]

    comparer = filecmp.dircmp(outpath, srcpath, ignore=ignore)
    assert not comparer.left_only and not comparer.right_only

    with open(os.path.join(outpath, f"{setup}.py"), "r") as f:
        setup_content = f.read()

    assert "VEROS_VERSION" in setup_content
