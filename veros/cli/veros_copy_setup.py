#!/usr/bin/env python

import os
import shutil
import pkg_resources
import functools

import click

SETUPDIR = pkg_resources.resource_filename('veros', 'setup')
SETUPS = sorted([
    setup for setup in os.listdir(SETUPDIR)
    if os.path.isdir(os.path.join(SETUPDIR, setup))
    and not setup.startswith('_')
])
IGNORE_PATTERNS = ['__init__.py', '*.pyc', '__pycache__/']


def copy_setup(setup, to=None):
    """Copy a standard setup to another directory"""
    if to is None:
        to = os.path.join(os.getcwd(), setup)

    parent = os.path.dirname(os.path.realpath(to))

    if not os.path.exists(parent):
        os.makedirs(parent)

    ignore = shutil.ignore_patterns(*IGNORE_PATTERNS)
    shutil.copytree(
        os.path.join(SETUPDIR, setup), to, ignore=ignore
    )


@click.command('veros-copy-setup')
@click.argument('setup', type=click.Choice(SETUPS))
@click.option('--to', type=click.Path(dir_okay=False, file_okay=False), required=False,
              default=None, help='Target directory (default: copy to current working directory)')
@functools.wraps(copy_setup)
def cli(*args, **kwargs):
    copy_setup(*args, **kwargs)


if __name__ == '__main__':
    cli()
