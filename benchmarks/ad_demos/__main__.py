"""Run each acceptance experiment in a fresh process and retain failure evidence."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import DEMOS
from .matrix import CASES, RESPONSES, TASKS, manifest


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return None
    return value


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker(name, config, path):
    config = dict(config, artifact_directory=str(path.parent))
    start = time.monotonic()
    result = dict(demo=name, config=config, measured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    checks = None
    try:
        # These demos run serially, including in installations that provide MPI.
        try:
            import mpi4py
        except ImportError:
            pass
        else:
            mpi4py.rc.initialize = False
        import jax
        from veros import runtime_settings

        runtime_settings.mpi_comm = None
        runtime_settings.backend = "jax"
        runtime_settings.device = config["device"]
        runtime_settings.diskless_mode = True
        runtime_settings.linear_solver = "scipy_jax"
        runtime_settings.loglevel = "warning"
        jax.config.update("jax_enable_x64", runtime_settings.float_type == "float64")
        result.update(jax_version=jax.__version__, devices=[str(device) for device in jax.devices()])
        result.update(dtype=runtime_settings.float_type, use_special_tdma=runtime_settings.use_special_tdma)
        from .provenance import collect

        result["provenance"] = collect(config, source_directory=path.parent / "sources")
        from .common import Checks

        def checkpoint_result():
            result.update(
                status="running",
                checks=checks.items,
                metrics=checks.metrics,
                seconds=time.monotonic() - start,
            )
            write_result(path, result)

        checks = Checks(on_update=checkpoint_result)
        checkpoint_result()
        if name in CASES:
            from .research import run
        elif name == "bathymetry_optimization":
            from .bathymetry import run
        elif name in ("closure_learning", "equilibrium_sensitivity"):
            from .column import run
        else:
            from .ocean import run
        run(name, config, checks)
        passed = bool(checks.items) and all(item["passed"] for item in checks.items)
        result["status"] = (
            ("smoke_passed" if passed else "smoke_failed")
            if config["mode"] == "smoke"
            else ("passed" if passed else "failed")
        )
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    if checks is not None:
        result.update(checks=checks.items, metrics=checks.metrics)
    result["seconds"] = time.monotonic() - start
    try:
        import resource
    except ImportError:
        pass
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_mib"] = peak / (1024**2 if sys.platform == "darwin" else 1024)
    write_result(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("all", "matrix", *DEMOS), default="all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="print machine-readable capability and question coverage")
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--response", choices=tuple(RESPONSES))
    parser.add_argument("--setup", choices=("acc", "global_4deg", "adjusted_basin"), default="acc")
    parser.add_argument("--mode", choices=("smoke", "study", "acceptance"), default="smoke")
    parser.add_argument("--physics", choices=("default", "fixed"), default="default")
    parser.add_argument("--neutral-policy", choices=("default", "nonstable"), default="default")
    parser.add_argument("--tke-policy", choices=("default", "positive_diffusion"))
    parser.add_argument("--ram-checkpoints", type=int, default=4)
    parser.add_argument("--disk-checkpoints", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--spinup-steps", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("ad-demo-results"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.matrix:
        print(json.dumps(manifest(), indent=2))
        return 0
    if args.list:
        print("\n".join(f"{name}: {description}" for name, description in DEMOS.items()))
        return 0
    if (args.steps is not None and args.steps < 1) or args.spinup_steps < 0 or args.timeout <= 0:
        parser.error("steps and timeout must be positive; spinup-steps must be nonnegative")
    if args.ram_checkpoints < 0 or args.disk_checkpoints < 0 or args.ram_checkpoints + args.disk_checkpoints < 1:
        parser.error("checkpoint counts must be nonnegative with at least one slot")
    config = {
        key: getattr(args, key)
        for key in (
            "setup",
            "mode",
            "physics",
            "device",
            "steps",
            "spinup_steps",
            "neutral_policy",
            "ram_checkpoints",
            "disk_checkpoints",
        )
    }
    if args.tke_policy is not None:
        config["tke_policy"] = args.tke_policy
    if args.worker:
        if args.demo in ("all", "matrix"):
            parser.error("a worker requires one demo")
        worker(args.demo, config, args.output)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    names = tuple(DEMOS) if args.demo == "all" else (tuple(CASES) if args.demo == "matrix" else (args.demo,))
    if args.task or args.response:
        names = tuple(
            name
            for name in names
            if name in CASES
            and (not args.task or CASES[name].task == args.task)
            and (not args.response or CASES[name].response == args.response)
        )
        if not names:
            parser.error("no matrix demos match this selection")
    results = []
    for name in names:
        result_path, log_path = args.output / f"{name}.json", args.output / f"{name}.log"
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "benchmarks.ad_demos",
            "--worker",
            "--demo",
            name,
            "--output",
            str(result_path),
        ]
        for key, value in config.items():
            if value is not None:
                command.extend(("--" + key.replace("_", "-"), str(value)))
        environment = dict(os.environ, VEROS_BACKEND="jax", VEROS_DEVICE=args.device)
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("OPENBLAS_NUM_THREADS", "1")
        environment["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
        print(f"Running {name} ({args.mode}, {args.setup})", flush=True)
        with log_path.open("w") as log:
            try:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=args.timeout
                )
                if process.returncode or not result_path.exists():
                    interrupted = (
                        json.loads(result_path.read_text()) if result_path.exists() else dict(demo=name, config=config)
                    )
                    interrupted.update(status="error", returncode=process.returncode, log=str(log_path))
                    write_result(result_path, interrupted)
            except subprocess.TimeoutExpired:
                interrupted = (
                    json.loads(result_path.read_text()) if result_path.exists() else dict(demo=name, config=config)
                )
                interrupted.update(status="timeout", timeout_seconds=args.timeout, log=str(log_path))
                write_result(result_path, interrupted)
        result = json.loads(result_path.read_text())
        results.append(result)
        failed = [item["name"] for item in result.get("checks", []) if not item["passed"]]
        print(f"{name}: {result['status']}" + (f" ({', '.join(failed)})" if failed else ""), flush=True)
    write_result(args.output / "summary.json", dict(config=config, results=results))
    required_status = "passed" if args.mode == "acceptance" else "smoke_passed"
    return int(any(result["status"] != required_status for result in results))


if __name__ == "__main__":
    sys.exit(main())
