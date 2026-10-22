"""Opt-in response profiling and inverse-problem diagnostics."""

import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from .__main__ import write_result
from .common import Checks
from .matrix import CASES


def configure(device="cpu"):
    try:
        import mpi4py
    except ImportError:
        pass
    else:
        mpi4py.rc.initialize = False
    from veros import runtime_settings
    import jax

    runtime_settings.mpi_comm = None
    runtime_settings.backend = "jax"
    runtime_settings.device = device
    runtime_settings.diskless_mode = True
    runtime_settings.linear_solver = "scipy_jax"
    runtime_settings.loglevel = "warning"
    jax.config.update("jax_enable_x64", True)


def profile(config, report, save):
    """Measure existing terminal checkpoint AD separately from the matrix adapter."""
    import jax
    import jax.numpy as jnp
    from veros.checkpointing import BinomialSchedule, CheckpointedIntegrator
    from .research_ocean import CONTROL_NAMES, ResearchOcean

    if config["neutral_policy"] == "surface":
        import importlib

        neutral = importlib.import_module("veros.core.isoneutral.isoneutral")
        divide = neutral.stable_divide

        def controlled_divide(numerator, denominator):
            # Restrict the surface-only diagnostic control to the 15-level setup.
            if numerator.shape[-1] == 15:
                numerator = numerator.at[..., -1].set(0)
                denominator = denominator.at[..., -1].set(-1)
            return divide(numerator, denominator)

        neutral.stable_divide = controlled_divide

    checks = Checks(on_update=save)
    report.update(checks=checks.items, metrics=checks.metrics, stage="setup")
    save()
    started = time.perf_counter()
    ocean = ResearchOcean(config, CASES["sensitivity_trajectory"])
    prepared = ocean.prepared
    checks.metrics.update(
        setup_seconds=time.perf_counter() - started,
        settings=dict(ocean.sim.state.settings.items()),
        prepared_time_seconds=float(ocean.initial.variables.time),
        controls=[
            CONTROL_NAMES[index]
            for index in ((2, 3, 4, 5, 6, 7) if config["control_family"] == "matrix_forcing" else (0, 2, 4, 8, 9))
        ],
        horizon_days=config["steps"] * ocean.dt / 86400,
        preparation_days=config["spinup_steps"] * ocean.dt / 86400,
        scope="terminal finite-window derivative with fixed preparation",
        matrix_adapter="terminal checkpoint driver measured independently of matrix observation windows",
    )
    save()
    indices = jnp.array([2, 3, 4, 5, 6, 7] if config["control_family"] == "matrix_forcing" else [0, 2, 4, 8, 9])
    point = jnp.zeros(indices.size)
    direction = jnp.arange(1, point.size + 1, dtype=point.dtype)
    if config["control_family"] not in ("joint", "matrix_forcing"):
        direction = jnp.asarray(
            {"initial": [1, 0, 0, 0, 0], "forcing": [0, 1, 1, 0, 0], "parameters": [0, 0, 0, 1, 1]}[
                config["control_family"]
            ],
            dtype=point.dtype,
        )
    direction /= jnp.linalg.norm(direction)
    checks.metrics.update(control_point=point.tolist(), control_direction=direction.tolist())
    initial_vs = ocean.initial.variables
    wet = initial_vs.maskW[2:-2, 2:-2, :-1]
    stratification = initial_vs.Nsqr[2:-2, 2:-2, :-1, initial_vs.tau]
    checks.metrics.update(
        initial_negative_wet_faces=int(jnp.sum(wet & (stratification < 0))),
        initial_nonpositive_wet_faces=int(jnp.sum(wet & (stratification <= 0))),
    )

    def controls(z):
        return ocean.controls(jnp.zeros(len(CONTROL_NAMES)).at[indices].set(z))

    feature = ocean.heat_index if config["objective"] == "heat" else ocean.amoc_index

    def terminal(carry):
        if config["objective"] == "combined":
            return jnp.sum(
                ocean.observe(carry)[
                    jnp.asarray((ocean.heat_index, ocean.heat_transport_index, ocean.acc_index, ocean.amoc_index))
                ]
            )
        return ocean.observe(carry)[feature]

    contraction = None
    if "checkpoint" in config["products"]:
        report["stage"] = "checkpoint"
        save()
        (carry, parameters), pullback = jax.vjp(controls, point)
        carry, parameters = controls(point)
        started = time.perf_counter()
        driver = CheckpointedIntegrator(prepared, terminal_objective=terminal)
        result = driver.value_and_grad(
            carry,
            parameters=parameters,
            n_steps=config["steps"],
            schedule=BinomialSchedule(config["steps"], ram_checkpoints=config.get("ram_checkpoints", 4)),
        )
        gradient = pullback((result.gradient, result.parameter_gradient))[0]
        jax.block_until_ready(gradient)
        contraction = float(jnp.vdot(gradient, direction))
        checks.metrics.update(
            checkpoint_seconds=time.perf_counter() - started,
            checkpoint=result.statistics,
            value=float(result.value),
            gradient=gradient.tolist(),
            directional_adjoint=contraction,
            final_failure_code=int(result.final.failure_code),
        )
        vs = result.final.variables
        checks.metrics["unstable_wet_faces"] = int(
            jnp.sum((vs.Nsqr[2:-2, 2:-2, :-1, vs.tau] < 0) & vs.maskW[2:-2, 2:-2, :-1])
        )
        checks.require("finite checkpoint gradient", np.isfinite(gradient).all(), gradient.tolist())
        del driver, result, pullback, carry, parameters
        jax.clear_caches()
        gc.collect()
        save()

    def step(c, p):
        return prepared.step(c, parameters=p)[0]

    if "tangent" in config["products"]:
        report["stage"] = "tangent"
        save()
        tangent_step = jax.jit(lambda c, p, dc, dp: jax.jvp(step, (c, p), (dc, dp)))
        started = time.perf_counter()
        (carry, parameters), (dc, dp) = jax.jvp(controls, (point,), (direction,))
        carry, parameters = controls(point)
        for _ in range(config["steps"]):
            _, dc = tangent_step(carry, parameters, dc, dp)
            carry = step(carry, parameters)
            prepared.check_status(carry)
        _, tangent = jax.jvp(terminal, (carry,), (dc,))
        value = terminal(carry)
        jax.block_until_ready(tangent)
        checks.metrics.update(tangent_seconds=time.perf_counter() - started, directional_tangent=float(tangent))
        if contraction is not None:
            checks.close("checkpoint/tangent value", value, checks.metrics["value"], rtol=1e-10)
            checks.close("checkpoint/tangent derivative", tangent, contraction)
        checks.metrics["tangent_value"] = float(value)
        checks.require("finite tangent", np.isfinite(tangent), float(tangent))
        contraction = float(tangent)
        del tangent_step, carry, parameters, dc, dp
        jax.clear_caches()
        gc.collect()
        save()

    def forward(z, *, record=False):
        c, p = controls(z)
        c, _ = prepared.advance(c, parameters=p, n_steps=config["steps"], final_only=True)
        prepared.check_status(c)
        if record:
            vs = c.variables
            stratification = vs.Nsqr[2:-2, 2:-2, :-1, vs.tau]
            checks.metrics.update(
                final_negative_wet_faces=int(jnp.sum(wet & (stratification < 0))),
                final_nonpositive_wet_faces=int(jnp.sum(wet & (stratification <= 0))),
            )
        return terminal(c)

    if "finite_difference" in config["products"]:
        report["stage"] = "finite_difference"
        save()
        started = time.perf_counter()
        differences = []
        baseline = float(forward(point, record=True))
        checks.metrics.update(
            forward_value=baseline,
            initial_objective=float(terminal(ocean.initial)),
            forward_steps=(1 + 2 * len(config["fd_epsilons"])) * config["steps"],
        )
        checks.require("finite forward value", np.isfinite(baseline), baseline)
        for epsilon in config["fd_epsilons"]:
            plus = forward(point + epsilon * direction)
            minus = forward(point - epsilon * direction)
            derivative = float((plus - minus) / (2 * epsilon))
            differences.append(dict(epsilon=epsilon, derivative=derivative))
            if contraction is not None:
                checks.close(f"finite difference {epsilon}", derivative, contraction, rtol=2e-3, atol=1e-8)
                checks.require(
                    f"finite update prediction {epsilon}",
                    abs(float(plus) - baseline - epsilon * contraction) <= 1e-8 + 5e-3 * abs(epsilon * contraction),
                    dict(actual_change=float(plus) - baseline, predicted_change=epsilon * contraction),
                )
            else:
                checks.require(f"finite difference {epsilon} finite", np.isfinite(derivative), derivative)
        checks.metrics.update(finite_difference_seconds=time.perf_counter() - started, finite_differences=differences)
        checks.metrics["compiled_timestep_signatures"] = prepared._compiled_step._cache_size()
        if contraction is None:
            checks.close(
                "finite-difference scale agreement",
                differences[0]["derivative"],
                differences[1]["derivative"],
                rtol=2e-3,
                atol=1e-8,
            )
            for first, second in zip(differences[1:-1], differences[2:]):
                checks.close(
                    f"finite-difference scales {first['epsilon']}/{second['epsilon']}",
                    first["derivative"],
                    second["derivative"],
                    rtol=2e-3,
                    atol=1e-8,
                )
        save()


def diagnose(config, report, save):
    """Repeat the existing optimizer with fixed objectives and increased budgets."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from .common import descend
    from .research import control_indices, jacobian, mean_feature
    from .research_ocean import ResearchOcean
    from .responses import Response

    case = CASES[config["case"]]
    ocean = ResearchOcean(config, case)
    response = Response(ocean, control_indices(case), config)
    point = jnp.zeros(response.indices.size)
    report["metrics"] = dict(
        prepared_time_seconds=float(ocean.initial.variables.time),
        runs=[],
        settings=dict(ocean.sim.state.settings.items()),
    )
    metrics = report["metrics"]
    rng = np.random.default_rng(42)
    if case.task in ("estimation", "calibration"):
        truth = jnp.linspace(0.15, 0.4, point.size)
        target = response.values(truth)
        baseline = response.values(point)
        scale = jnp.maximum(jnp.abs(target - baseline), 1e-3)
        nz = ocean.nz
        train = (
            jnp.arange(4 * nz + 4)
            if response.kind == "local"
            else jnp.concatenate((jnp.arange(2 * nz), jnp.array([ocean.heat_transport_index, ocean.amoc_index])))
        )

        def loss(z):
            residual = (response.values(z)[..., train] - target[..., train]) / scale[..., train]
            return jnp.mean(residual**2) + 5e-4 * jnp.sum(z**2)

        matrix = jacobian(lambda z: response.values(z)[..., train], point)
        _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        informed = singular > max(1e-6, singular[0] * 1e-6)
        metrics.update(
            truth=truth.tolist(),
            singular_values=singular.tolist(),
            numerical_rank=int(informed.sum()),
            loss_at_truth=float(loss(truth)),
            truth_control_norm=float(jnp.linalg.norm(truth)),
            objective_scope="same baseline scaling, prior, training observations, and unit ball for every start",
        )
        for label, initial, iterations in (
            ("zero_24", point, 24),
            ("zero_96", point, 96),
            ("random_96", jnp.asarray(rng.normal(0, 0.1, point.size)), 96),
            ("truth_96", truth, 96),
        ):
            optimizer = {}
            fit, history = descend(loss, initial, iterations=iterations, radius=1, diagnostics=optimizer)
            metrics["runs"].append(
                dict(
                    label=label,
                    fitted=fit.tolist(),
                    history=history,
                    optimizer=optimizer,
                    observable_error=float(np.linalg.norm(vt[informed] @ np.asarray(fit - truth))),
                    raw_recovery_error=float(jnp.linalg.norm(fit - truth)),
                )
            )
            save()
    elif case.task == "perturbation":
        baseline = float(mean_feature(response.values(point), ocean.amoc_index, response.kind))
        target = 0.9 * baseline

        def slack(z):
            return target - mean_feature(response.values(z), ocean.amoc_index, response.kind)

        derivative = jax.grad(slack)
        directions = rng.normal(size=(16, point.size))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        samples = np.array([float(slack(jnp.asarray(z))) for z in directions])
        gradient = np.asarray(derivative(point))
        metrics.update(
            baseline_sv=baseline,
            target_sv=target,
            sampled_slack=samples.tolist(),
            sampled_directions=directions.tolist(),
            baseline_gradient=gradient.tolist(),
            linearized_best_slack=float(slack(point)) + np.linalg.norm(gradient),
            interpretation="local searches and sampled controls do not certify global infeasibility",
        )
        for label, start in (
            ("zero", np.zeros(point.size)),
            ("best_sample", directions[np.argmax(samples)]),
            ("gradient", gradient / max(np.linalg.norm(gradient), 1e-30)),
        ):
            result = minimize(
                lambda z: -float(slack(jnp.asarray(z))),
                start,
                jac=lambda z: -np.asarray(derivative(jnp.asarray(z))),
                method="SLSQP",
                constraints=dict(type="ineq", fun=lambda z: 1 - z @ z, jac=lambda z: -2 * z),
                options=dict(maxiter=96, ftol=1e-12),
            )
            metrics["runs"].append(
                dict(
                    label=label,
                    success=bool(result.success),
                    message=str(result.message),
                    iterations=int(result.nit),
                    fitted=result.x.tolist(),
                    slack=float(slack(jnp.asarray(result.x))),
                    control_norm=float(np.linalg.norm(result.x)),
                )
            )
            save()
    elif case.task == "observation":

        def candidates(z):
            return response.values(z)[..., : 4 * ocean.nz].reshape(-1)

        def target(z):
            return mean_feature(response.values(z, heldout=True), ocean.amoc_index, response.kind)

        matrix = jacobian(candidates, point)
        gradient = np.asarray(jax.grad(target)(point))
        covariance = np.linalg.inv(np.eye(point.size) + matrix.T @ matrix)
        singular = np.linalg.svd(matrix, compute_uv=False)
        metrics.update(
            candidate_singular_values=singular.tolist(),
            future_transport_gradient=gradient.tolist(),
            prior_variance=float(gradient @ gradient),
            all_candidate_variance=float(gradient @ covariance @ gradient),
            all_candidate_fraction=float(gradient @ covariance @ gradient / max(gradient @ gradient, 1e-30)),
            interpretation="all-candidate covariance is a local Gaussian information limit for the stated candidate set",
        )
        save()


def worker(config, destination):
    import resource
    from .provenance import collect

    started = time.perf_counter()
    report = dict(
        config=config, status="running", measured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def save():
        report.update(
            seconds=time.perf_counter() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
        write_result(destination, report)

    save()
    try:
        configure(config.get("device", "cpu"))
        report["provenance"] = collect(config, source_directory=destination.parent / "sources")
        save()
        (profile if config["lane"] == "profile" else diagnose)(config, report, save)
        report["status"] = "measured"
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    save()


def guarded(command, destination, *, timeout, memory_mib, reserve_mib):
    """Retain incremental measurements when a resource budget stops a worker."""
    import psutil

    destination.unlink(missing_ok=True)
    started, peak, reason = time.monotonic(), 0, None
    with destination.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                rss = sum(
                    child.memory_info().rss
                    for child in [parent, *parent.children(recursive=True)]
                    if child.is_running()
                )
            except psutil.Error:
                rss = 0
            peak = max(peak, rss)
            if rss > memory_mib * 2**20:
                reason = "process_memory_limit"
            elif psutil.virtual_memory().available < reserve_mib * 2**20:
                reason = "available_memory_reserve"
            elif time.monotonic() - started > timeout:
                reason = "timeout"
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.2)
    report = json.loads(destination.read_text()) if destination.exists() else {}
    if reason or process.returncode:
        report.update(status="resource_stop" if reason else "error", stop_reason=reason, returncode=process.returncode)
    report["resource_budget"] = dict(
        timeout_seconds=timeout,
        process_memory_mib=memory_mib,
        available_reserve_mib=reserve_mib,
        sampled_peak_process_tree_mib=peak / 2**20,
        seconds=time.monotonic() - started,
    )
    write_result(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("profile", "diagnose"), required=True)
    parser.add_argument(
        "--global", dest="global_run", action="store_true", help="explicitly enable expensive global profiles"
    )
    parser.add_argument("--physics", choices=("default", "fixed"), default="default")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 30], help="response timestep counts")
    parser.add_argument("--preparation", nargs="+", type=int, default=[0, 30], help="fixed preparation timestep counts")
    parser.add_argument(
        "--objectives", nargs="+", choices=("heat", "transport", "combined"), default=["heat", "transport"]
    )
    parser.add_argument(
        "--neutral-policy",
        choices=("default", "surface", "nonstable"),
        default="default",
        help="neutral policy; surface is a process-local diagnostic control",
    )
    parser.add_argument(
        "--control-family", choices=("joint", "initial", "forcing", "parameters", "matrix_forcing"), default="joint"
    )
    parser.add_argument("--fd-epsilons", nargs="+", type=float, default=[1e-3, 1e-4])
    parser.add_argument(
        "--products",
        nargs="+",
        choices=("checkpoint", "tangent", "finite_difference"),
        default=["checkpoint", "tangent", "finite_difference"],
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=[
            "estimation_local",
            "estimation_trajectory",
            "calibration_local",
            "calibration_trajectory",
            "perturbation_local",
            "perturbation_trajectory",
            "observation_local",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--memory-mib", type=int, default=5500)
    parser.add_argument("--reserve-mib", type=int, default=3000)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(json.loads(args.worker_config), args.output)
        return 0
    if any(value < 1 for value in args.horizons) or any(value < 0 for value in args.preparation):
        parser.error("horizons must be positive and preparation must be nonnegative")
    if args.timeout <= 0 or args.memory_mib <= 0 or args.reserve_mib < 0:
        parser.error("resource budgets must be positive; memory reserve may be zero")
    if (
        len(set(args.fd_epsilons)) != len(args.fd_epsilons)
        or len(args.fd_epsilons) < 2
        or any(not np.isfinite(value) or value <= 0 for value in args.fd_epsilons)
    ):
        parser.error("provide at least two distinct finite positive finite-difference perturbations")
    args.output.mkdir(parents=True, exist_ok=True)
    base = dict(
        mode="smoke",
        setup="global_4deg" if args.global_run else "acc",
        physics=args.physics,
        device="cpu",
        spinup_steps=0,
        steps=None,
        lane=args.lane,
    )
    jobs = []
    if args.lane == "profile":
        for horizon in args.horizons:
            for preparation in args.preparation:
                for objective in args.objectives:
                    config = dict(
                        base,
                        steps=horizon,
                        spinup_steps=preparation,
                        objective=objective,
                        products=args.products,
                        neutral_policy=args.neutral_policy,
                        control_family=args.control_family,
                        fd_epsilons=args.fd_epsilons,
                    )
                    jobs.append(
                        (
                            f"{base['setup']}-{args.physics}-{objective}-h{horizon}-p{preparation}-"
                            + "-".join(args.products)
                            + (f"-{args.neutral_policy}" if args.neutral_policy != "default" else "")
                            + (f"-{args.control_family}" if args.control_family != "joint" else "")
                            + ("-fdscan" if args.fd_epsilons != [1e-3, 1e-4] else ""),
                            config,
                        )
                    )
    else:
        for name in args.cases:
            if CASES[name].response not in ("local", "trajectory") or CASES[name].task not in (
                "estimation",
                "calibration",
                "perturbation",
                "observation",
            ):
                parser.error("diagnosis supports finite inverse, perturbation, and observation cases")
            jobs.append((name, dict(base, case=name)))
    results = []
    for name, config in jobs:
        print(f"Running {name}", flush=True)
        destination = args.output / f"{name}.json"
        destination.unlink(missing_ok=True)
        result = guarded(
            [
                sys.executable,
                "-m",
                "benchmarks.ad_demos.baseline",
                "--lane",
                args.lane,
                "--worker-config",
                json.dumps(config),
                "--output",
                str(destination),
            ],
            destination,
            timeout=args.timeout,
            memory_mib=args.memory_mib,
            reserve_mib=args.reserve_mib,
        )
        results.append(result)
        write_result(args.output / "summary.json", dict(results=results))
        print(f"{name}: {result['status']}", flush=True)
    return int(any(result["status"] != "measured" for result in results))


if __name__ == "__main__":
    sys.exit(main())
