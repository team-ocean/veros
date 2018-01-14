#!/usr/bin/env python

import os
import shutil
import pkg_resources

import veros.scripts.veros_copy_setup
import veros.scripts.veros_create_mask
import veros.scripts.veros_resubmit

import click

SETUPDIR = pkg_resources.resource_filename("veros", "setup")
SETUPS = sorted([setup for setup in os.listdir(SETUPDIR) if os.path.isdir(os.path.join(SETUPDIR, setup))])


@click.group()
def main():
    pass


@main.command("copy-setup")
@click.argument("setup", type=click.Choice(SETUPS))
@click.option("--target-dir", type=click.Path(exists=True, file_okay=False), required=False, default=None)
def copy_setup(setup, target_dir=None):

    if target_dir is None:
        target_dir = os.getcwd()

    shutil.copytree(os.path.join(SETUPDIR, setup), os.path.join(target_dir, setup))


@main.command()
def resubmit():
    pass


if __name__ == "__main__":
    main()
