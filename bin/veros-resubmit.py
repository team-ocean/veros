#!/usr/bin/env python

"""Performs several runs of Veros back to back, using the previous run as restart input.

Intended to be used with scheduling systems (e.g. SLURM or PBS).
"""

import argparse
import subprocess
import shlex
import sys
import os

LAST_N_FILENAME = "current_run.veros"

def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("IDENTIFIER", help="base identifier of the simulation")
    parser.add_argument("N_RUNS", type=int, help="total number of runs")
    parser.add_argument("LENGTH_PER_RUN", type=float, help="length (in seconds) of each run")
    parser.add_argument("VEROS_CMD", type=shlex.split, help="the command that is used to call veros (in quotes)")
    parser.add_argument("--callback", metavar="CMD", type=shlex.split,
                        default=[sys.executable, __file__] + sys.argv[1:],
                        help="command to call after each run has finished (defaults to this script)")
    return parser.parse_args()

def get_current_n():
    if not os.path.isfile(LAST_N_FILENAME):
        return 0
    with open(LAST_N_FILENAME, "r") as f:
        return int(f.read())

def write_next_n(n):
    with open(LAST_N_FILENAME, "w") as f:
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

if __name__ == "__main__":
    args = parse_cli()

    current_n = get_current_n()
    if current_n >= args.N_RUNS:
        sys.exit(0)

    call_veros(args.VEROS_CMD, args.IDENTIFIER, current_n, args.LENGTH_PER_RUN)
    write_next_n(current_n + 1)

    if current_n >= args.N_RUNS:
        sys.exit(0)

    if args.callback:
        subprocess.Popen(args.callback)
    else:
        subprocess.Popen()
