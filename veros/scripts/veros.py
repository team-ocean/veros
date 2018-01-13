#!/usr/bin/env python

import os
import shutil
import pkg_resources

try:
    import click
    have_click = True
except ImportError:
    have_click = False

if not have_click:
    raise ImportError("The Veros command line tools require click (e.g. through `pip install click`)")

SCRIPTDIR = os.path.realpath(os.path.dirname(__file__))
setupdir = pkg_resources.resource_filename(__name__, "setup/acc")
print(setupdir)
raise
SETUPDIR = os.path.realpath(os.path.join(SCRIPTDIR, "../../setup"))
SETUPS = [setup for setup in os.listdir(SETUPDIR) if os.path.isdir(os.path.join(SETUPDIR, setup))]


@click.group()
def main():
    pass


@main.command()
@click.argument("setup", choices=SETUPS)
@click.option("target-dir", type=click.Path(), default=None)
def copy_setup(setup, target_dir=None):
    if target_dir is None:
        target_dir = os.getcwd()
    shutil.copytree(os.path.join(SETUPDIR, setup), os.path.join(target_dir, setup))


@main.command()
def resubmit():
    pass


if __name__ == "__main__":
    main()
