#!/usr/bin/env python

import functools
import subprocess
import shlex
import pipes
import sys
import os
import time

import click

LAST_N_FILENAME = '{identifier}.current_run'
CHILD_TIMEOUT = 10
POLL_DELAY = 0.1


class ShellCommand(click.ParamType):
    name = 'command'

    def convert(self, value, param, ctx):
        return shlex.split(value)


def get_current_n(filename):
    if not os.path.isfile(filename):
        return 0

    with open(filename, 'r') as f:
        return int(f.read())


def write_next_n(n, filename):
    with open(filename, 'w') as f:
        f.write(str(n))


def unparse(args):
    return ' '.join(map(pipes.quote, args))


def call_veros(cmd, name, n, runlen):
    identifier = '{name}.{n:0>4}'.format(name=name, n=n)
    prev_id = '{name}.{n:0>4}'.format(name=name, n=n - 1)
    args = ['-s', 'identifier', identifier, '-s', 'restart_output_filename',
            '{identifier}.restart.h5', '-s', 'runlen', '{}'.format(runlen)]
    if n:
        args += ['-s', 'restart_input_filename', '{prev_id}.restart.h5'.format(prev_id=prev_id)]
    sys.stdout.write('\n >>> {}\n\n'.format(' '.join(cmd + args)))
    sys.stdout.flush()
    try:
        subprocess.check_call(unparse(cmd + args), shell=True)
    except subprocess.CalledProcessError:
        raise RuntimeError('Run {} failed, exiting'.format(n))


def resubmit(identifier, num_runs, length_per_run, veros_cmd, callback):
    """Performs several runs of Veros back to back, using the previous run as restart input.

    Intended to be used with scheduling systems (e.g. SLURM or PBS).

    """
    last_n_filename = LAST_N_FILENAME.format(identifier=identifier)

    current_n = get_current_n(last_n_filename)
    if current_n >= num_runs:
        return

    call_veros(veros_cmd, identifier, current_n, length_per_run)
    write_next_n(current_n + 1, last_n_filename)
    next_proc = subprocess.Popen(unparse(callback), shell=True)

    # catch immediately crashing processes
    timeout = CHILD_TIMEOUT

    while timeout > 0:
        retcode = next_proc.poll()
        if retcode is not None:
            if retcode > 0:
                # process crashed
                raise RuntimeError('Callback exited with {}'.format(retcode))
            else:
                break
        time.sleep(POLL_DELAY)
        timeout -= POLL_DELAY


@click.command('veros-resubmit', short_help='Re-run a Veros setup several times')
@click.option('-i', '--identifier', required=True,
              help='Base identifier of the simulation')
@click.option('-n', '--num-runs', type=click.INT, required=True,
              help='Total number of runs to execute')
@click.option('-l', '--length-per-run', type=click.FLOAT, required=True,
              help='Length (in seconds) of each run')
@click.option('-c', '--veros-cmd', type=ShellCommand(), required=True,
              help='The command that is used to call veros (quoted)')
@click.option('--callback', metavar='CMD', type=ShellCommand(), default=None,
              help='Command to call after each run has finished (quoted, default: call self)')
@functools.wraps(resubmit)
def cli(*args, **kwargs):
    if kwargs['callback'] is None:
        kwargs['callback'] = sys.argv

    resubmit(*args, **kwargs)


if __name__ == '__main__':
    cli()
