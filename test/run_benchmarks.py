#! /usr/bin/env python

import sys
import os
import subprocess
import multiprocessing
import time
import argparse
import imp
import math
import re
import json

"""
Runs all Veros benchmarks back to back and writes timing results to JSON outfile.
"""
testdir = os.path.join(os.path.dirname(__file__), os.path.relpath("benchmarks"))
nproc = multiprocessing.cpu_count()
benchmark_commands = {
    "numpy": "{python} {filename} -b numpy -s nx {nx} -s ny {ny} -s nz {nz}",
    "bohrium": "BH_STACK=openmp BH_OPENMP_PROF=1 {python} {filename} -b bohrium -s nx {nx} -s ny {ny} -s nz {nz}",
    "bohrium-opencl": "BH_STACK=opencl BH_OPENCL_PROF=1 {python} {filename} -b bohrium -s nx {nx} -s ny {ny} -s nz {nz}",
    "fortran": "{python} {filename} --fortran-lib {fortran_lib} -s nx {nx} -s ny {ny} -s nz {nz}",
    "fortran-mpi": "mpiexec -n {nproc} -- {python} {filename} --fortran-lib {fortran_lib} -s nx {nx} -s ny {ny} -s nz {nz}"
}
outfile = "benchmark_{}.json".format(time.time())

def parse_cli():
    parser = argparse.ArgumentParser(description="Run Veros benchmarks")
    parser.add_argument("-f", "--fortran-library", type=str)
    parser.add_argument("-s", "--max-size", type=float, required=True)
    parser.add_argument("-c", "--components", nargs="*", choices=benchmark_commands.keys(), default=["numpy"])
    return parser.parse_args()

def check_fortran_library(path):
    flib = None
    try:
        imp.load_dynamic("pyOM_code", args.fortran_library)
        flib = "sequential"
    except ImportError:
        pass
    try:
        imp.load_dynamic("pyOM_code_MPI", args.fortran_library)
        flib = "parallel"
    except ImportError:
        pass
    return flib

if __name__ == "__main__":
    args = parse_cli()
    fortran_version = check_fortran_library(args.fortran_library)

    if fortran_version is None and ("fortran" in args.components or "fortran-mpi" in args.components):
        raise RuntimeError("Path to fortran library must be giving when running fortran components")

    if fortran_version != "parallel" and "fortran-mpi" in args.components:
        raise RuntimeError("Fortran library must be compiled with MPI support for fortran-mpi component")

    sizes = [10 ** n for n in range(3,int(math.log10(args.max_size)))]
    out_data = {}
    try:
        for f in os.listdir(testdir):
            if not f.endswith("_benchmark.py"):
                continue

            out_data[f] = []
            print("running benchmark {} ".format(f))
            for size in sizes:
                n = int(size ** (1./3.)) + 1
                nx = ny = int(2. ** 0.5 * n)
                nz = n / 2
                print(" current size: {}".format(nx * ny * nz))
                cmd_args = {
                            "python": sys.executable,
                            "filename": os.path.realpath(os.path.join(testdir, f)),
                            "nproc": nproc,
                            "fortran_lib": args.fortran_library,
                            "nx": nx, "ny": ny, "nz": nz
                           }

                for comp in args.components:
                    cmd = benchmark_commands[comp]
                    sys.stdout.write("  {:<15} ... ".format(comp))
                    sys.stdout.flush()
                    start = time.time()
                    try: # must run each benchmark in its own Python subprocess to reload the Fortran library
            	        output = subprocess.check_output(cmd.format(**cmd_args),
                                                         shell=True,
                                                         stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        print("failed")
                        print(e.output)
                        continue
                    end = time.time()
                    elapsed = end - start
                    print("{:.2f}s".format(elapsed))

                    detailed_stats = None
                    if "bohrium" in comp:
                        detailed_stats = {"Pre-fusion": None, "Fusion": None,
                                          "Compile": None, "Exec": None}
                        patterns = {stat: r"\s*{}\:\s*(\d+\.?\d*)s".format(stat) for stat in detailed_stats.keys()}
                        for line in output.splitlines():
                            for stat in detailed_stats.keys():
                                match = re.match(patterns[stat], line)
                                if match:
                                    detailed_stats[stat] = float(match.group(1))

                    out_data[f].append({"component": comp,
                                        "size": nx * ny * nz,
                                        "wall_time": elapsed,
                                        "detailed_stats": detailed_stats})

    finally:
        with open(outfile, "w") as f:
            json.dump(out_data, f)
