import tempfile
import os
import filecmp
import fnmatch
import pkg_resources

from click.testing import CliRunner
import pytest

import veros.cli


@pytest.fixture(scope='module')
def runner():
    return CliRunner()


def test_veros_copy_setup(runner):
    with tempfile.TemporaryDirectory() as tempdir:
        for setup in ('acc', 'acc_basic', 'acc_sector', 'global_4deg', 'global_1deg',
                      'global_flexible', 'north_atlantic', 'wave_propagation'):
            result = runner.invoke(veros.cli.veros_copy_setup.cli, [setup, '--to', os.path.join(tempdir, setup)])
            assert result.exit_code == 0, setup
            assert not result.output

            outpath = os.path.join(tempdir, setup)
            srcpath = pkg_resources.resource_filename('veros', 'setup/%s' % setup)
            ignore = [f for f in os.listdir(srcpath) if any(
                fnmatch.fnmatch(f, pattern) for pattern in veros.cli.veros_copy_setup.IGNORE_PATTERNS
            )]
            ignore.append('version.txt')

            comparer = filecmp.dircmp(outpath, srcpath, ignore=ignore)
            assert not comparer.left_only and not comparer.right_only and not comparer.diff_files
