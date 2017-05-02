import sys
import os
import subprocess
import multiprocessing
import time

"""
Runs all Veros benchmarks back to back.
Expects path to the compiled pyOM2 Fortran library as first command line argument
(preferably with MPI support).
"""

try:
    fortran_path = sys.argv[1]
except IndexError:
    raise RuntimeError("First command line argument must be path to pyOM Fortran library")

sizes = [10 ** n for n in range(4,7)]
nproc = multiprocessing.cpu_count()
benchmark_commands = (
                      "python {filename} -b numpy -s nx {nx} -s ny {ny} -s nz {nz}",
                      "BH_STACK=openmp python {filename} -b bohrium -s nx {nx} -s ny {ny} -s nz {nz}",
                      "BH_STACK=opencl python {filename} -b bohrium -s nx {nx} -s ny {ny} -s nz {nz}",
                      "python {filename} -f {fortran_lib} -s nx {nx} -s ny {ny} -s nz {nz}",
                      "mpirun -n {nproc} -- python {filename} -f {fortran_lib} -s nx {nx} -s ny {ny} -s nz {nz}"
                      )
outfile = "benchmark_{}.log".format(time.time())

with open(outfile, "w") as log:
    for f in os.listdir("./benchmarks"):
        if f.endswith("_benchmark.py"):
            print("Running test {} ... ".format(f))
            for size in sizes:
                n = int(size ** (1./3.)) + 1
                nx = ny = int(2. ** 0.5 * n)
                nz = n / 2
                args = {
                            "filename": os.path.realpath(os.path.join("./benchmarks", f)),
                            "nproc": nproc,
                            "fortran_lib": fortran_path,
                            "nx": nx, "ny": ny, "nz": nz
                       }
                for cmd in benchmark_commands:

                    print(" > " + cmd.format(**args))
                    start = time.time()
                    try: # must run each benchmark in its own Python subprocess to reload the Fortran library
            	        output = subprocess.check_output(cmd.format(**args), shell=True,
                                                          stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        print(" failed")
                        print(e.output)
                        continue
                    end = time.time()
                    elapsed = end - start
                    print(" finished after {}s".format(elapsed))
                    log.write("{}".format(elapsed))
