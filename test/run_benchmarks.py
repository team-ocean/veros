#! /usr/bin/env python

import sys
import os
import subprocess
import multiprocessing
import time
import argparse
import imp
import math

"""
Runs all Veros benchmarks back to back.
Expects path to the compiled pyOM2 Fortran library as first command line argument
(with MPI support).
"""

nproc = multiprocessing.cpu_count()
benchmark_commands = {
    "numpy": "{python} {filename} -b numpy -s nx {nx} -s ny {ny} -s nz {nz}",
    "bohrium": "BH_STACK=openmp {python} {filename} -b bohrium -s nx {nx} -s ny {ny} -s nz {nz}",
    "bohrium-opencl": "BH_STACK=opencl {python} {filename} -b bohrium -s nx {nx} -s ny {ny} -s nz {nz}",
    "fortran": "{python} {filename} --fortran-lib {fortran_lib} -s nx {nx} -s ny {ny} -s nz {nz}",
    "fortran-mpi": "mpiexec -n {nproc} -- {python} {filename} --fortran-lib {fortran_lib} -s nx {nx} -s ny {ny} -s nz {nz}"
}
outfile = "benchmark_{}.log".format(time.time())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Veros benchmarks")
    parser.add_argument("-f", "--fortran-library", type=str)
    parser.add_argument("-s", "--max-size", type=float, required=True)
    parser.add_argument("-c", "--components", nargs="*", choices=benchmark_commands.keys(), default=["numpy"])
    args = parser.parse_args()

    if args.fortran_library:
        try:
            imp.load_dynamic("pyOM_code", args.fortran_library)
            has_fortran_mpi = False
        except ImportError:
            imp.load_dynamic("pyOM_code_MPI", args.fortran_library)
            has_fortran_mpi = True

    if not has_fortran_mpi and "fortran-mpi" in args.components:
        raise RuntimeError("Fortran library must be compiled with MPI support for fortran-mpi component")

    sizes = [10 ** n for n in range(3,int(math.log10(args.max_size)),1)]

    with open(outfile, "w") as log:
        log.write("Size, Component, Time")
        for f in os.listdir("./benchmarks"):
            if f.endswith("_benchmark.py"):
                print("Running test {} ... ".format(f))
                for size in sizes:
                    n = int(size ** (1./3.)) + 1
                    nx = ny = int(2. ** 0.5 * n)
                    nz = n / 2
                    cmd_args = {
                                "python": sys.executable,
                                "filename": os.path.realpath(os.path.join("./benchmarks", f)),
                                "nproc": nproc,
                                "fortran_lib": args.fortran_library,
                                "nx": nx, "ny": ny, "nz": nz
                               }
                    for comp in args.components:
                        cmd = benchmark_commands[comp]
                        print(" > " + cmd.format(**cmd_args))
                        start = time.time()
                        try: # must run each benchmark in its own Python subprocess to reload the Fortran library
                	        output = subprocess.check_output(cmd.format(**cmd_args),
                                                             shell=True,
                                                             stderr=subprocess.STDOUT)
                        except subprocess.CalledProcessError as e:
                            print(" failed")
                            print(e.output)
                            continue
                        end = time.time()
                        elapsed = end - start
                        print(" finished after {}s".format(elapsed))
                        log.write("{}, {}, {}".format(nx * ny * nz, comp, elapsed))
