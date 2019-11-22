#!/usr/bin/env python

import os
import shutil
import functools

import click
import entrypoints


SETUPDIR_ENVVAR = 'VEROS_SETUP_DIR'
IGNORE_PATTERNS = ['__init__.py', '*.pyc', '__pycache__/']
SETUPS = {}

setup_dirs = [
    os.path.dirname(e.load().__file__)
    for e in entrypoints.get_group_all('veros.setup_dirs')
]

for setup_dir in os.environ.get(SETUPDIR_ENVVAR, '').split(';'):
    if os.path.isdir(setup_dir):
        setup_dirs.append(setup_dir)

# populate {setup_name: path} mapping
for setup_dir in setup_dirs:
    for setup in os.listdir(setup_dir):
        setup_path = os.path.join(setup_dir, setup)
        if not os.path.isdir(setup_path):
            continue
        if setup.startswith(('_', '.')):
            continue
        SETUPS[setup] = setup_path

SETUP_NAMES = sorted(SETUPS.keys())


def write_version_file(target_dir, origin):
    from veros import __version__ as veros_version

    with open(os.path.join(target_dir, 'version.txt'), 'w') as f:
        f.write(
            'Veros v{veros_version}\n'
            '{origin}\n'
            .format(origin=origin, veros_version=veros_version)
        )


def copy_setup(setup, to=None):
    """Copy a standard setup to another directory.

    Available setups:

        {setups}

    Example:

        $ veros copy-setup global_4deg --to ~/veros-setups/4deg-lowfric

    Further directories containing setup templates can be added to this command
    via the {setup_envvar} environment variable.
    """
    if to is None:
        to = os.path.join(os.getcwd(), setup)

    if os.path.exists(to):
        raise RuntimeError('Target directory must not exist')

    to_parent = os.path.dirname(os.path.realpath(to))

    if not os.path.exists(to_parent):
        os.makedirs(to_parent)

    ignore = shutil.ignore_patterns(*IGNORE_PATTERNS)
    shutil.copytree(
        SETUPS[setup], to, ignore=ignore
    )

    write_version_file(to, SETUPS[setup])


copy_setup.__doc__ = copy_setup.__doc__.format(
    setups=', '.join(SETUP_NAMES), setup_envvar=SETUPDIR_ENVVAR
)


@click.command('veros-copy-setup')
@click.argument('setup', type=click.Choice(SETUP_NAMES), metavar='SETUP')
@click.option('--to', required=False, default=None,
              type=click.Path(dir_okay=False, file_okay=False, writable=True),
              help=('Target directory, must not exist '
                    '(default: copy to current working directory)'))
@functools.wraps(copy_setup)
def cli(*args, **kwargs):
    copy_setup(*args, **kwargs)


if __name__ == '__main__':
    cli()
