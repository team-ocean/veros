import os
import click


def benchmark_cli(func):
    @click.option("--size", type=int, nargs=3, required=True)
    @click.option("--timesteps", type=int, required=True)
    @click.option("-f", "--pyom2-lib", type=click.Path(readable=True, dir_okay=False), default=None)
    @click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="numpy")
    @click.option("-d", "--device", type=click.Choice(["cpu", "gpu"]), default="cpu")
    @click.option("-n", "--nproc", type=int, nargs=2, default=(1, 1))
    @click.option("--float-type", type=click.Choice(["float64", "float32"]), default="float64")
    @click.option("-v", "--loglevel", type=click.Choice(["debug", "trace"]), default="debug")
    @click.option("--profile-mode", is_flag=True)
    @click.command()
    def inner(backend, device, nproc, float_type, loglevel, profile_mode, linear_solver, petsc_options, **kwargs):
        from veros import runtime_settings, runtime_state

        print(petsc_options)
        runtime_settings.update(
            backend=backend,
            device=device,
            float_type=float_type,
            num_proc=nproc,
            loglevel=loglevel,
            profile_mode=profile_mode,
        )

        if device == "gpu" and runtime_state.proc_num > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime_state.proc_rank)

        return func(**kwargs)

    return inner
