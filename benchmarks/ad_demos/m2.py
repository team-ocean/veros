"""Guarded adjusted-response and closure studies."""

import argparse
from pathlib import Path
import sys

from .baseline import guarded
from .__main__ import write_result
from .matrix import CASES

M2_CASES = tuple(
    name
    for name, case in CASES.items()
    if case.task != "geometry"
    and (
        case.response in ("equilibrium", "seasonal")
        or (case.task == "closure" and case.response in ("local", "trajectory"))
    )
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("all", *M2_CASES), default="all")
    parser.add_argument("--setup", choices=("global_4deg", "adjusted_basin", "acc"), default="adjusted_basin")
    parser.add_argument("--physics", choices=("default", "fixed"), default="fixed")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--neutral-policy", choices=("default", "nonstable"), default="nonstable")
    parser.add_argument(
        "--task", choices=("sensitivity", "estimation", "calibration", "perturbation", "closure", "observation")
    )
    parser.add_argument("--response", choices=("local", "trajectory", "equilibrium", "seasonal"))
    parser.add_argument("--spinup-steps", type=int, default=0)
    parser.add_argument("--ram-checkpoints", type=int)
    parser.add_argument("--timeout", type=float, default=43200)
    parser.add_argument("--memory-mib", type=int, default=6500)
    parser.add_argument("--reserve-mib", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/optimization-m2"))
    args = parser.parse_args()
    if (
        args.spinup_steps < 0
        or (args.ram_checkpoints is not None and args.ram_checkpoints < 1)
        or args.timeout <= 0
        or args.memory_mib <= 0
        or args.reserve_mib < 0
    ):
        parser.error("Invalid preparation or resource budget")
    ram_checkpoints = args.ram_checkpoints
    if ram_checkpoints is None:
        ram_checkpoints = 64 if args.setup == "adjusted_basin" else 4
    names = M2_CASES if args.demo == "all" else (args.demo,)
    names = tuple(
        name
        for name in names
        if (not args.task or CASES[name].task == args.task)
        and (not args.response or CASES[name].response == args.response)
    )
    if not names:
        parser.error("No adjusted-response or closure cases match this selection")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in names:
        path = args.output / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.ad_demos",
            "--worker",
            "--demo",
            name,
            "--setup",
            args.setup,
            "--physics",
            args.physics,
            "--device",
            args.device,
            "--neutral-policy",
            args.neutral_policy,
            "--mode",
            "acceptance" if args.setup == "global_4deg" else "study",
            "--spinup-steps",
            str(args.spinup_steps),
            "--ram-checkpoints",
            str(ram_checkpoints),
            "--output",
            str(path),
        ]
        print("Starting", name, flush=True)
        result = guarded(command, path, timeout=args.timeout, memory_mib=args.memory_mib, reserve_mib=args.reserve_mib)
        reports.append(
            dict(
                case=name,
                path=str(path),
                status=result.get("status"),
                failed_checks=[check["name"] for check in result.get("checks", []) if not check["passed"]],
                resource_budget=result.get("resource_budget"),
            )
        )
        write_result(args.output / "summary.json", dict(setup=args.setup, physics=args.physics, results=reports))
        print("Finished", name, result.get("status"), flush=True)
    return int(any(report["status"] != "passed" for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
