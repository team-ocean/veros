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
    parser.add_argument("VEROS_CMD", help="the command that is used to call veros")
    return parser.parse_args()

def call_veros(cmd, name, n, runlen):
    identifier = "{name}.{n:0>4}".format(name=name, n=n)
    prev_id = "{name}.{n:0>4}".format(name=name, n=n-1)
    args = "-s identifier \"{name}\" -s restart_output_filename \"{{identifier}}.restart.h5\" -s runlen {runlen}" \
            .format(name=identifier, runlen=runlen)
    if n:
        args += " -s restart_input_filename \"{prev_id}.restart.h5\"".format(prev_id=prev_id)
    sys.stdout.write(" >>> " + cmd + " " + args + "\n")
    sys.stdout.flush()
    subprocess.check_call(shlex.split(cmd) + shlex.split(args))

if __name__ == "__main__":
    args = parse_cli()
    for n in range(args.N_SUBMITS):
        call_veros(args.VEROS_CMD, args.IDENTIFIER, n, args.LENGTH_PER_RUN)
