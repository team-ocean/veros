#!/usr/bin/env python

import functools
import subprocess
import shlex
import sys
import os
import time

import click

LAST_N_FILENAME = "{identifier}.current_run"


class ShellCommand(click.ParamType):
    def convert(self, value, param, ctx):
        return shlex.split(value)


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("IDENTIFIER", help="base identifier of the simulation")
    parser.add_argument("N_RUNS", type=int, help="total number of runs")
    parser.add_argument("LENGTH_PER_RUN", type=float, help="length (in seconds) of each run")
    parser.add_argument("VEROS_CMD", type=shlex.split, help="the command that is used to call veros (quoted)")
    parser.add_argument("--callback", )
    return parser.parse_args()


def get_current_n(filename):
    if not os.path.isfile(filename):
        return 0
    with open(filename, "r") as f:
        return int(f.read())


def write_next_n(n, filename):
    with open(filename, "w") as f:
        f.write(str(n))


def call_veros(cmd, name, n, runlen):
    identifier = "{name}.{n:0>4}".format(name=name, n=n)
    prev_id = "{name}.{n:0>4}".format(name=name, n=n-1)
    args = ["-s", "identifier", identifier, "-s", "restart_output_filename", "{identifier}.restart.h5", "-s", "runlen", "{}".format(runlen)]
    if n:
        args += ["-s", "restart_input_filename", "{prev_id}.restart.h5".format(prev_id=prev_id)]
    sys.stdout.write("\n >>> {}\n\n".format(" ".join(cmd + args)))
    sys.stdout.flush()
    try:
        subprocess.check_call(cmd + args)
    except subprocess.CalledProcessError:
        raise RuntimeError("Run {} failed, exiting".format(n))


def resubmit(identifier, n_runs, length_per_run, veros_cmd, callback=None):
    """Performs several runs of Veros back to back, using the previous run as restart input.

    Intended to be used with scheduling systems (e.g. SLURM or PBS).

    """
    last_n_filename = LAST_N_FILENAME.format(identifier=identifier)

    current_n = get_current_n(last_n_filename)
    if current_n >= args.N_RUNS:
        sys.exit(0)

    call_veros(args.VEROS_CMD, args.IDENTIFIER, current_n, args.LENGTH_PER_RUN)
    write_next_n(current_n + 1, last_n_filename)

    if current_n >= args.N_RUNS:
        sys.exit(0)

    subprocess.Popen(args.callback)
    time.sleep(30) # make sure next process is properly spawned before exiting


@click.command("veros-resubmit", short_help="Re-run a Veros setup several times")
@click.argument("identifier")
@click.option("--callback", metavar="CMD", type=ShellCommand(),
              default=[sys.executable, __file__] + sys.argv[1:],
              help="Command to call after each run has finished (default: call self)")
@functools.wraps(resubmit)
def cli(*args, **kwargs):
    pass


if __name__ == "__main__":
    cli()
