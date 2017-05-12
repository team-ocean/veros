#!/usr/bin/env python

"""Performs several runs of Veros back to back, using the previous run as restart input.

Intended to be used with scheduling systems (e.g. SLURM or PBS).
"""

import argparse
import subprocess
import shlex
import sys

def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("IDENTIFIER", help="base identifier of the simulation")
    parser.add_argument("N_SUBMITS", type=int, help="total number of resubmits")
    parser.add_argument("LENGTH_PER_RUN", type=float, help="length (in seconds) of each run")
    parser.add_argument("VEROS_CMD", type=shlex.split, help="the command that is used to call veros (in quotes)")
    parser.add_argument("-n", "--start-n", metavar="N", type=int, default=0, help="re-start from resubmit run with number N")
    return parser.parse_args()

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
    for n in range(args.N_SUBMITS):
        call_veros(args.VEROS_CMD, args.IDENTIFIER, n + args.start_n, args.LENGTH_PER_RUN)
