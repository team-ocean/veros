#! /usr/bin/env python

import sys
import os
import subprocess
import multiprocessing
import argparse
import imp
import math
import re
import json
import time
import numpy as np

"""
Runs selected Veros benchmarks back to back and writes timing results to a JSON file.
"""

TESTDIR = os.path.join(os.path.dirname(__file__), os.path.relpath("benchmarks"))
COMPONENTS = ["numpy", "bohrium", "bohrium-opencl", "bohrium-cuda", "fortran", "fortran-mpi"]
STATIC_SETTINGS = "-v debug -s nx {nx} -s ny {ny} -s nz {nz} -s default_float_type {float_type} --timesteps {timesteps}"
BENCHMARK_COMMANDS = {
    "numpy": "OMP_NUM_THREADS={nproc} {python} {filename} -b numpy " + STATIC_SETTINGS,
    "bohrium": "OMP_NUM_THREADS={nproc} BH_STACK=openmp BH_OPENMP_PROF=1 {python} {filename} -b bohrium "  + STATIC_SETTINGS,
    "bohrium-opencl": "BH_STACK=opencl BH_OPENCL_PROF=1 {python} {filename} -b bohrium " + STATIC_SETTINGS,
    "bohrium-cuda": "BH_STACK=cuda BH_CUDA_PROF=1 {python} {filename} -b bohrium " + STATIC_SETTINGS,
    "fortran": "{python} {filename} --fortran-lib {fortran_lib} " + STATIC_SETTINGS,
    "fortran-mpi": "{mpiexec} -n {nproc} --allow-run-as-root -- {python} {filename} --fortran-lib {fortran_lib} " + STATIC_SETTINGS
}
SLURM_COMMANDS = {
    "numpy": "OMP_NUM_THREADS={nproc} srun --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b numpy " + STATIC_SETTINGS,
    "bohrium": "OMP_NUM_THREADS={nproc} BH_STACK=openmp BH_OPENMP_PROF=1 srun --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b bohrium " + STATIC_SETTINGS,
    "bohrium-opencl": "BH_STACK=opencl BH_OPENCL_PROF=1 srun --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b bohrium " + STATIC_SETTINGS,
    "bohrium-cuda": "BH_STACK=cuda BH_CUDA_PROF=1 srun --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b bohrium " + STATIC_SETTINGS,
    "fortran": "srun --ntasks 1 -- {python} {filename} --fortran-lib {fortran_lib} " + STATIC_SETTINGS,
    "fortran-mpi": "srun --ntasks {nproc} --cpus-per-task 1 -- {python} {filename} --fortran-lib {fortran_lib} " + STATIC_SETTINGS
}
AVAILABLE_BENCHMARKS = [f for f in os.listdir(TESTDIR) if f.endswith("_benchmark.py")]

def parse_cli():
    parser = argparse.ArgumentParser(description="Run Veros benchmarks")
    parser.add_argument("-f", "--fortran-library", type=str, help="Path to pyOM2 fortran library")
    parser.add_argument("-s", "--sizes", nargs="*", type=float, required=True,
                        help="Problem sizes to test (total number of elements)")
    parser.add_argument("-c", "--components", nargs="*", choices=COMPONENTS, default=["numpy"], metavar="COMPONENT",
                        help="Numerical backend components to benchmark (possible values: {})".format(", ".join(COMPONENTS)))
    parser.add_argument("-n", "--nproc", type=int, default=multiprocessing.cpu_count(),
                        help="Number of processes / threads for parallel execution")
    parser.add_argument("-o", "--outfile", default="benchmark_{}.json".format(time.time()), help="JSON file to write timings to")
    parser.add_argument("-t", "--timesteps", default=1000, type=int, help="Number of time steps that each benchmark is run for")
    parser.add_argument("--only", nargs="*", default=AVAILABLE_BENCHMARKS,
                        help="Run only these benchmarks (possible values: {})".format(", ".join(AVAILABLE_BENCHMARKS)),
                        choices=AVAILABLE_BENCHMARKS, required=False, metavar="BENCHMARK")
    parser.add_argument("--mpiexec", default="mpiexec", help="Executable used for calling MPI (e.g. mpirun, mpiexec)")
    parser.add_argument("--slurm", action="store_true", help="Run benchmarks using SLURM scheduling command (srun)")
    parser.add_argument("--debug", action="store_true", help="Additionally print each command that is executed")
    parser.add_argument("--float-type", default="float64", help="Data type for floating point arrays in Veros components")
    parser.add_argument("--burnin", default=3, type=int, help="Number of iterations to exclude in timings")
    return parser.parse_args()

def check_arguments(args):
    fortran_version = check_fortran_library(args.fortran_library)

    if not fortran_version and ("fortran" in args.components or "fortran-mpi" in args.components):
        raise RuntimeError("Path to fortran library must be given when running fortran components")
    if fortran_version != "parallel" and "fortran-mpi" in args.components:
        raise RuntimeError("Fortran library must be compiled with MPI support for fortran-mpi component")
    if args.float_type != "float64" and ("fortran" in args.components or "fortran-mpi" in args.components):
        raise RuntimeError("Can run Fortran components only with 'float64' float type")
    if not args.burnin < args.timesteps:
        raise RuntimeError("burnin must be smaller than number of timesteps")

def check_fortran_library(path):
    if not path:
        return None
    try:
        imp.load_dynamic("pyOM_code", args.fortran_library)
        return "sequential"
    except ImportError:
        pass
    try:
        imp.load_dynamic("pyOM_code_MPI", args.fortran_library)
        return "parallel"
    except ImportError:
        pass
    return None

if __name__ == "__main__":
    args = parse_cli()
    check_arguments(args)

    settings = {
        "timesteps": args.timesteps,
        "nproc": args.nproc,
        "float_type": args.float_type,
        "burnin": args.burnin
    }
    out_data = {}
    all_passed = True
    try:
        for f in args.only:
            out_data[f] = []
            print("running benchmark {} ".format(f))
            for size in args.sizes:
                n = int(size ** (1./3.)) + 1
                nx = ny = int(2. ** 0.5 * n)
                nz = n // 2
                real_size = nx * ny * nz
                print(" current size: {}".format(real_size))
                cmd_args = {
                            "python": sys.executable,
                            "filename": os.path.realpath(os.path.join(TESTDIR, f)),
                            "nproc": args.nproc,
                            "mpiexec": args.mpiexec,
                            "fortran_lib": args.fortran_library,
                            "nx": nx, "ny": ny, "nz": nz,
                            "timesteps": args.timesteps,
                            "float_type": args.float_type
                           }

                for comp in args.components:
                    cmd = (SLURM_COMMANDS[comp] if args.slurm else BENCHMARK_COMMANDS[comp]).format(**cmd_args)
                    if args.debug:
                        print("  $ " + cmd)
                    sys.stdout.write("  {:<15} ... ".format(comp))
                    sys.stdout.flush()
                    try: # must run each benchmark in its own Python subprocess to reload the Fortran library
            	        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        print("failed")
                        print(e.output)
                        all_passed = False
                        continue
                    iteration_times = list(map(float, re.findall(r"Time step took ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)s", output)))[args.burnin:]
                    if not iteration_times:
                        raise RuntimeError("could not extract iteration times from output")
                    total_elapsed = sum(iteration_times)
                    print("{:>6.2f}s".format(total_elapsed))

                    bohrium_stats = None
                    if "bohrium" in comp:
                        bohrium_stats = {"Pre-fusion": None, "Fusion": None,
                                         "Compile": None, "Exec": None,
                                         "Copy2dev": None, "Copy2host": None,
                                         "Offload": None, "Other": None}
                        patterns = {stat: r"\s*{}\:\s*(\d+\.?\d*)s".format(stat) for stat in bohrium_stats.keys()}
                        for line in output.splitlines():
                            for stat in bohrium_stats.keys():
                                match = re.match(patterns[stat], line)
                                if match:
                                    bohrium_stats[stat] = float(match.group(1))

                    out_data[f].append({
                        "component": comp,
                        "size": real_size,
                        "wall_time": total_elapsed,
                        "per_iteration": {
                            "best": np.min(iteration_times),
                            "worst": np.max(iteration_times),
                            "mean": np.mean(iteration_times),
                            "stdev": np.std(iteration_times),
                        },
                        "bohrium_stats": bohrium_stats
                    })
    finally:
        with open(args.outfile, "w") as f:
            json.dump({"benchmarks": out_data, "settings": settings}, f, indent=4)

    sys.exit(int(not all_passed))
