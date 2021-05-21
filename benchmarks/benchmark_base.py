import click


def benchmark_cli(func):
    @click.option("--size", type=int, nargs=3, required=True)
    @click.option("--timesteps", type=int, required=True)
    @click.option("-f", "--pyom2-lib", type=click.Path(readable=True, dir_okay=False), default=None)
    @click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="numpy")
    @click.option("-d", "--device", type=click.Choice(["cpu", "gpu"]), default="cpu")
    @click.option("-n", "--nproc", type=int, nargs=2, default=(1, 1))
    @click.option("--float-type", type=click.Choice(["float64", "float32"]), default="float64")
    @click.command()
    def inner(backend, device, nproc, float_type, **kwargs):
        from veros import runtime_settings

        runtime_settings.update(
            loglevel="debug",
            backend=backend,
            device=device,
            float_type=float_type,
            num_proc=nproc,
        )
        return func(**kwargs)

    return inner
