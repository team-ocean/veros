import tempfile
import os
import shutil
import contextlib
import filecmp
import fnmatch
import pkg_resources

from click.testing import CliRunner
import pytest

import veros.cli


@contextlib.contextmanager
def TemporaryDirectory():
    tempdir = tempfile.mkdtemp()
    try:
        yield tempdir
    finally:
        shutil.rmtree(tempdir)


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


def test_veros_copy_setup(runner):
    with TemporaryDirectory() as tempdir:
        for setup in ("acc", "eady", "global_4deg", "global_1deg", "north_atlantic", "wave_propagation"):
            result = runner.invoke(veros.cli.veros_copy_setup.cli, [setup, "--to", tempdir])
            assert result.exit_code == 0
            assert not result.output

            outpath = os.path.join(tempdir, setup)
            srcpath = pkg_resources.resource_filename("veros", "setup/%s" % setup)
            ignore = [f for f in os.listdir(srcpath) if any(
                fnmatch.fnmatch(f, pattern) for pattern in veros.cli.veros_copy_setup.IGNORE_PATTERNS
            )]
            comparer = filecmp.dircmp(outpath, srcpath, ignore=ignore)
            assert not comparer.left_only and not comparer.right_only and not comparer.diff_files
