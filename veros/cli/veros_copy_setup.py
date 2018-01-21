#!/usr/bin/env python

import os
import shutil
import pkg_resources
import functools

import click

SETUPDIR = pkg_resources.resource_filename("veros", "setup")
SETUPS = sorted([setup for setup in os.listdir(SETUPDIR) if os.path.isdir(os.path.join(SETUPDIR, setup))])
IGNORE_PATTERNS = ["__init__.py", "*.pyc", "__pycache__/"]


def copy_setup(setup, target_dir=None):
    """Copy a standard setup to another directory"""
    if target_dir is None:
        target_dir = os.getcwd()

    ignore = shutil.ignore_patterns(*IGNORE_PATTERNS)
    shutil.copytree(
        os.path.join(SETUPDIR, setup), os.path.join(target_dir, setup), ignore=ignore
    )


@click.command("veros-copy-setup")
@click.argument("setup", type=click.Choice(SETUPS))
@click.option("--target-dir", type=click.Path(exists=True, file_okay=False), required=False,
              default=None, help="Target directory (defaults to current directory)")
@functools.wraps(copy_setup)
def cli(*args, **kwargs):
    copy_setup(*args, **kwargs)
