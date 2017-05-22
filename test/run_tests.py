#! /usr/bin/env python

import argparse
import sys
import os
import subprocess

"""
Runs all veros tests back to back and compares the results with the legacy (pyOM2) backend.
Expects path to the compiled Fortran library as first command line argument.
"""

testdir = os.path.join(os.path.dirname(__file__), os.path.relpath("tests"))
available_tests = [f for f in os.listdir(testdir) if f.endswith("_test.py")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Veros testing suite (requires pyOM)")
    parser.add_argument("pyomlib", type=os.path.realpath, help="Path to pyOM fortran library to test against")
    parser.add_argument("--only", nargs="*", default=available_tests, help="Run only these tests", choices=available_tests, required=False)
    parser.add_argument("--no-bohrium", action="store_true", help="Disable testing with Bohrium", required=False)
    args = parser.parse_args()

    success, fail = "passed", "failed"
    if sys.stdout.isatty():
        success = "\x1b[{}m{}\x1b[0m".format("32",success)
        fail = "\x1b[{}m{}\x1b[0m".format("31",fail)

    all_passed = True
    for testscript in args.only:
        sys.stdout.write("Running test {} ... ".format(testscript))
        sys.stdout.flush()
        try: # must run each test in its own Python subprocess to reload the Fortran library
	        output = subprocess.check_output([sys.executable, os.path.join(testdir, testscript), args.pyomlib],
                                              stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            sys.stdout.write(fail + "\n\n")
            print(e.output)
            all_passed = False
            continue
        sys.stdout.write(success + "\n")

        if args.no_bohrium:
            continue

        sys.stdout.write("Running test {} with Bohrium ... ".format(f))
        sys.stdout.flush()
        try: # must run each test in its own Python subprocess to reload the Fortran library
	        output = subprocess.check_output([sys.executable, os.path.join(testdir, testscript), args.pyomlib, "-b", "bohrium"],
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            sys.stdout.write(fail + "\n\n")
            print(e.output)
            all_passed = False
            continue
        sys.stdout.write(success + "\n")
    sys.exit(int(not all_passed))
