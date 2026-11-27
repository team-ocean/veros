"""Opt-in finite-response verification with explicit resource limits."""

import argparse
import datetime
from pathlib import Path
import sys
import time
import traceback

from .__main__ import write_result
from .baseline import configure, guarded


def worker(config, destination):
    from .provenance import collect
    from .feasibility import (
        statistical_benchmarks,
        reference_data,
        ocean_feasibility,
        physics_profile,
        jacobian_profile,
    )

    configure(config.get("device", "cpu"))
    report = dict(
        config=config, measured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status="running"
    )
    report["provenance"] = collect(config, source_directory=destination.parent / "sources")
    start = time.perf_counter()

    def save():
        report["seconds"] = time.perf_counter() - start
        write_result(destination, report)

    save()
    try:
        if config["lane"] == "statistics":
            statistical_benchmarks(report, save)
        elif config["lane"] == "reference":
            reference_data(report, save, destination)
        elif config["lane"] == "ocean":
            ocean_feasibility(config, report, save)
        elif config["lane"] == "jacobian":
            jacobian_profile(config, report, save)
        else:
            physics_profile(config, report, save, destination)
        report["status"] = "passed" if report["checks"] and all(c["passed"] for c in report["checks"]) else "failed"
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane", choices=("matrix", "statistics", "reference", "ocean", "physics", "jacobian"), required=True
    )
    parser.add_argument("--setup", choices=("acc", "global_4deg"), default="acc")
    parser.add_argument("--physics", choices=("default", "fixed"), default="default")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--neutral-policy", choices=("default", "nonstable"), default="default")
    parser.add_argument("--tke-policy", choices=("default", "positive_diffusion"))
    parser.add_argument("--mode", choices=("smoke", "acceptance"), default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--spinup-steps", type=int, default=0)
    parser.add_argument("--ram-checkpoints", type=int, default=4)
    parser.add_argument("--disk-checkpoints", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--memory-mib", type=float, default=6500)
    parser.add_argument("--reserve-mib", type=float, default=1800)
    parser.add_argument("--comparison-report", type=Path, help="reuse verified individual columns in the Jacobian lane")
    parser.add_argument("--jacobian-repeats", type=int, default=1, help="measure repeated grouped Jacobian calls")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0 or args.memory_mib <= 0 or args.reserve_mib < 0 or args.spinup_steps < 0:
        parser.error("invalid resource or preparation budget")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    if args.jacobian_repeats < 1:
        parser.error("jacobian-repeats must be positive")
    if args.ram_checkpoints < 0 or args.disk_checkpoints < 0 or args.ram_checkpoints + args.disk_checkpoints == 0:
        parser.error("provide nonnegative checkpoint counts and at least one slot")
    if args.lane in ("physics", "jacobian") and args.steps is None:
        args.steps = 360 if args.setup == "global_4deg" else 96
    if args.lane == "ocean":
        if args.setup != "global_4deg":
            parser.error("ocean feasibility requires --setup global_4deg")
        args.steps = args.steps or 720
        if args.steps < 120:
            parser.error("ocean feasibility requires at least four monthly samples")
    config = {
        key: getattr(args, key)
        for key in (
            "lane",
            "setup",
            "physics",
            "device",
            "neutral_policy",
            "mode",
            "steps",
            "spinup_steps",
            "ram_checkpoints",
            "disk_checkpoints",
        )
    }
    if args.lane == "jacobian":
        config["jacobian_repeats"] = args.jacobian_repeats
    if args.comparison_report is not None:
        if args.lane != "jacobian":
            parser.error("comparison-report requires the Jacobian lane")
        config["comparison_report"] = str(args.comparison_report)
    if args.tke_policy is not None:
        config["tke_policy"] = args.tke_policy
    if args.worker:
        worker(config, args.output)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if args.lane == "matrix":
        from .matrix import CASES

        names = [
            name
            for name, case in CASES.items()
            if case.task in ("sensitivity", "estimation", "calibration", "perturbation", "observation")
            and case.response in ("local", "trajectory")
        ]
    else:
        names = [args.lane]
    reports = []
    for name in names:
        if args.lane == "matrix":
            command = [sys.executable, "-m", "benchmarks.ad_demos", "--worker", "--demo", name]
        else:
            command = [sys.executable, "-m", "benchmarks.ad_demos.m1", "--worker", "--lane", args.lane]
        for key, value in config.items():
            if key == "lane" or value is None or (key == "steps" and name.endswith("_local")):
                continue
            command.extend(["--" + key.replace("_", "-"), str(value)])
        destination = args.output / f"{name}.json"
        command.extend(["--output", str(destination)])
        print(f"Running {name}", flush=True)
        report = guarded(
            command, destination, timeout=args.timeout, memory_mib=args.memory_mib, reserve_mib=args.reserve_mib
        )
        reports.append(report)
        print(f"{name}: {report.get('status', 'error')}", flush=True)
    write_result(args.output / "summary.json", dict(config=config, results=reports))
    return int(any(report.get("status") not in ("passed", "smoke_passed") for report in reports))


if __name__ == "__main__":
    sys.exit(main())
