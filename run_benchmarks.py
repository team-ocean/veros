#! /usr/bin/env python

import sys
import os
import subprocess
import multiprocessing
import importlib.util
import re
import time
import math
import itertools
import json

import click
import numpy as np

"""
Runs selected Veros benchmarks back to back and writes timing results to a JSON file.
"""

TESTDIR = os.path.join(os.path.dirname(__file__), os.path.relpath("benchmarks"))

COMPONENTS = ["numpy", "numpy-mpi", "jax", "jax-gpu", "jax-mpi", "jax-gpu-mpi", "fortran", "fortran-mpi"]

STATIC_SETTINGS = " --size {nx} {ny} {nz} --timesteps {timesteps} --float-type {float_type}"

BENCHMARK_COMMANDS = {
    "numpy": "{python} {filename}" + STATIC_SETTINGS,
    "numpy-mpi": "OMP_NUM_THREADS=1 {mpiexec} -n {nproc} {python} {filename} --nproc {decomp}" + STATIC_SETTINGS,
    "jax": "{python} {filename} -b jax" + STATIC_SETTINGS,
    "jax-gpu": "{python} {filename} -b jax --device gpu" + STATIC_SETTINGS,
    "jax-mpi": "OMP_NUM_THREADS=1 {mpiexec} -n {nproc} {python} {filename} -b jax --nproc {decomp}" + STATIC_SETTINGS,
    "jax-gpu-mpi": "OMP_NUM_THREADS=1 {mpiexec} -n {nproc} {python} {filename} -b jax --device gpu --nproc {decomp}"
    + STATIC_SETTINGS,
    "fortran": "{python} {filename} --pyom2-lib {pyom2_lib}" + STATIC_SETTINGS,
    "fortran-mpi": "{mpiexec} -n {nproc} {python} {filename} --pyom2-lib {pyom2_lib} --nproc {decomp}"
    + STATIC_SETTINGS,
}
SLURM_COMMANDS = {
    "numpy": "{mpiexec} --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b numpy" + STATIC_SETTINGS,
    "numpy-mpi": "{mpiexec} --ntasks {nproc} --cpus-per-task 1 -- {python} {filename} -b numpy --nproc {decomp}"
    + STATIC_SETTINGS,
    "jax": "{mpiexec} --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b jax" + STATIC_SETTINGS,
    "jax-gpu": "{mpiexec} --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b jax --device gpu"
    + STATIC_SETTINGS,
    "jax-mpi": "{mpiexec} --ntasks {nproc} --cpus-per-task 1 -- {python} {filename} -b jax --nproc {decomp}"
    + STATIC_SETTINGS,
    "jax-gpu-mpi": "{mpiexec} --ntasks {nproc} --cpus-per-task 1 -- {python} {filename} -b jax --device gpu --nproc {decomp}"
    + STATIC_SETTINGS,
    "fortran": "{mpiexec} --ntasks 1 -- {python} {filename} --pyom2-lib {pyom2_lib}" + STATIC_SETTINGS,
    "fortran-mpi": "{mpiexec} --ntasks {nproc} --cpus-per-task 1 -- {python} {filename} --pyom2-lib {pyom2_lib} --nproc {decomp}"
    + STATIC_SETTINGS,
}

AVAILABLE_BENCHMARKS = [f for f in os.listdir(TESTDIR) if f.endswith("_benchmark.py")]

TIME_PATTERN = r"Time step took ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)s"


def check_arguments(pyom2_lib, components, float_type, burnin, timesteps, **kwargs):
    fortran_version = check_pyom2_lib(pyom2_lib)

    if "fortran" in components or "fortran-mpi" in components:
        if not pyom2_lib:
            raise click.UsageError("Path to fortran library must be given when running fortran components")

        if not fortran_version:
            raise click.UsageError("Fortran library failed to import")

    if fortran_version != "parallel" and "fortran-mpi" in components:
        raise click.UsageError("Fortran library must be compiled with MPI support for fortran-mpi component")

    if float_type != "float64" and ("fortran" in components or "fortran-mpi" in components):
        raise click.UsageError('Can run Fortran components only with "float64" float type')

    if not burnin < timesteps:
        raise click.UsageError("burnin must be smaller than number of timesteps")


def check_pyom2_lib(path):
    if not path:
        return None

    def _check_library(module):
        spec = importlib.util.spec_from_file_location(module, path)
        try:
            importlib.util.module_from_spec(spec)
        except ImportError:
            return False
        else:
            return True

    if _check_library("pyOM_code"):
        return "sequential"

    if _check_library("pyOM_code_MPI"):
        return "parallel"

    return None


def _factorize(num):
    j = 2
    while num > 1:
        for i in range(j, int(math.sqrt(num + 0.05)) + 1):
            if num % i == 0:
                num /= i
                j = i
                yield i
                break
        else:
            if num > 1:
                yield num
                break


def _decompose_num(num, into=2):
    out = [1] * into
    for fac, i in zip(_factorize(num), itertools.cycle(range(into))):
        out[i] *= fac

    return tuple(map(int, out))


def _round_to_multiple(num, divisor):
    return int(round(num / divisor) * divisor)


@click.command("veros-benchmarks", help="Run Veros benchmarks")
@click.option("-f", "--pyom2-lib", type=str, help="Path to PyOM2 fortran library")
@click.option(
    "-s", "--sizes", multiple=True, type=float, required=True, help="Problem sizes to test (total number of elements)"
)
@click.option(
    "-c",
    "--components",
    multiple=True,
    type=click.Choice(COMPONENTS),
    default=["numpy"],
    metavar="COMPONENT",
    help="Numerical backend components to benchmark (possible values: {})".format(", ".join(COMPONENTS)),
)
@click.option(
    "-n",
    "--nproc",
    type=int,
    default=multiprocessing.cpu_count(),
    help="Number of processes / threads for parallel execution",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    default="benchmark_{}.json".format(time.time()),
    help="JSON file to write timings to",
)
@click.option("-t", "--timesteps", default=100, type=int, help="Number of time steps that each benchmark is run for")
@click.option(
    "--only",
    multiple=True,
    default=AVAILABLE_BENCHMARKS,
    help="Run only these benchmarks (possible values: {})".format(", ".join(AVAILABLE_BENCHMARKS)),
    type=click.Choice(AVAILABLE_BENCHMARKS),
    required=False,
    metavar="BENCHMARK",
)
@click.option("--mpiexec", default=None, help="Executable used for calling MPI (e.g. mpirun, mpiexec)")
@click.option("--slurm", is_flag=True, help="Run benchmarks using SLURM scheduling command (srun)")
@click.option("--debug", is_flag=True, help="Additionally print each command that is executed")
@click.option("--float-type", default="float64", help="Data type for floating point arrays in Veros components")
@click.option("--burnin", default=3, type=int, help="Number of iterations to exclude in timings")
def run(**kwargs):
    check_arguments(**kwargs)

    proc_decom = _decompose_num(kwargs["nproc"], 2)

    settings = kwargs.copy()
    settings["decomp"] = f"{proc_decom[0]} {proc_decom[1]}"

    out_data = {}
    all_passed = True
    try:
        for f in kwargs["only"]:
            out_data[f] = []
            click.echo(f"running benchmark {f}")

            for size in kwargs["sizes"]:
                n = math.ceil(size ** (1 / 3))
                nz = max(n // 2, 2)
                nx = _round_to_multiple(math.sqrt(2) * n, proc_decom[0])
                ny = _round_to_multiple(math.sqrt(2) * n, proc_decom[1])
                real_size = nx * ny * nz

                click.echo(f" current size: {real_size}")

                cmd_args = settings.copy()
                cmd_args.update(
                    {
                        "python": sys.executable,
                        "filename": os.path.realpath(os.path.join(TESTDIR, f)),
                        "nx": nx,
                        "ny": ny,
                        "nz": nz,
                    }
                )

                if cmd_args["mpiexec"] is None:
                    if kwargs["slurm"]:
                        cmd_args["mpiexec"] = "srun"
                    else:
                        cmd_args["mpiexec"] = "mpirun"

                for comp in kwargs["components"]:
                    cmd = (SLURM_COMMANDS[comp] if kwargs["slurm"] else BENCHMARK_COMMANDS[comp]).format(**cmd_args)

                    if kwargs["debug"]:
                        click.echo(f"  $ {cmd}")

                    sys.stdout.write(f"  {comp:<15} ... ")
                    sys.stdout.flush()

                    try:  # must run each benchmark in its own Python subprocess to reload the Fortran library
                        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        click.echo("failed")
                        click.echo(e.output.decode("utf-8"))
                        all_passed = False
                        continue

                    output = output.decode("utf-8")
                    iteration_times = list(map(float, re.findall(TIME_PATTERN, output)))[kwargs["burnin"] :]
                    if not iteration_times:
                        raise RuntimeError("could not extract iteration times from output")

                    total_elapsed = sum(iteration_times)
                    click.echo(f"{total_elapsed:>6.2f}s")

                    out_data[f].append(
                        {
                            "component": comp,
                            "size": real_size,
                            "wall_time": total_elapsed,
                            "per_iteration": {
                                "best": float(np.min(iteration_times)),
                                "worst": float(np.max(iteration_times)),
                                "mean": float(np.mean(iteration_times)),
                                "stdev": float(np.std(iteration_times)),
                            },
                        }
                    )
    finally:
        with open(kwargs["outfile"], "w") as f:
            json.dump({"benchmarks": out_data, "settings": settings}, f, indent=4, sort_keys=True)

    raise SystemExit(int(not all_passed))


if __name__ == "__main__":
    run()
