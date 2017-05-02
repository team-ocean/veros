#! /usr/bin/env python

import sys
import os
import subprocess

"""
Runs all veros tests back to back and compares the results with the legacy (pyOM2) backend.
Expects path to the compiled Fortran library as first command line argument.
"""

if __name__ == "__main__":
    try:
        fortran_path = sys.argv[1]
    except IndexError:
        raise RuntimeError("First command line argument must be path to pyOM Fortran library")

    success, fail = "passed", "failed"
    if sys.stdout.isatty():
        success = "\x1b[{}m{}\x1b[0m".format("32",success)
        fail = "\x1b[{}m{}\x1b[0m".format("31",fail)

    for f in os.listdir("./tests"):
        if f.endswith("_test.py"):
            sys.stdout.write("Running test {} ... ".format(f))
            sys.stdout.flush()
            try: # must run each test in its own Python subprocess to reload the Fortran library
    	        output = subprocess.check_output(["python", os.path.join("./tests", f), fortran_path],
                                                  stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                sys.stdout.write(fail + "\n\n")
                print(e.output)
                continue
            sys.stdout.write(success + "\n")

            sys.stdout.write("Running test {} with Bohrium ... ".format(f))
            sys.stdout.flush()
            try: # must run each test in its own Python subprocess to reload the Fortran library
    	        output = subprocess.check_output(["python", os.path.join("./tests", f), fortran_path, "-b", "bohrium"],
                                                  stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                sys.stdout.write(fail + "\n\n")
                print(e.output)
                continue
            sys.stdout.write(success + "\n")
