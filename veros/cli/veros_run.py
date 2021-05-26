import functools
import inspect
import os
import sys
import importlib

import click

from veros import runtime_settings, VerosSetup, __version__ as veros_version
from veros.settings import SETTINGS
from veros.backend import BACKENDS
from veros.runtime import LOGLEVELS, DEVICES, FLOAT_TYPES


class VerosSetting(click.ParamType):
    name = "setting"
    current_key = None

    def convert(self, value, param, ctx):
        assert param.nargs == 2

        if self.current_key is None:
            if value not in SETTINGS:
                self.fail(f"Unknown setting {value}")
            self.current_key = value
            return value

        assert self.current_key in SETTINGS
        setting = SETTINGS[self.current_key]
        self.current_key = None

        if setting.type is bool:
            return click.BOOL(value)

        return setting.type(value)


def _import_from_file(path):
    module = os.path.basename(path).split(".py")[0]
    spec = importlib.util.spec_from_file_location(module, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(setup_file, *args, **kwargs):
    """Runs a Veros setup from given file"""
    kwargs["override"] = dict(kwargs["override"])

    runtime_setting_kwargs = (
        "backend",
        "profile_mode",
        "num_proc",
        "loglevel",
        "device",
        "float_type",
        "diskless_mode",
        "force_overwrite",
    )
    for setting in runtime_setting_kwargs:
        setattr(runtime_settings, setting, kwargs.pop(setting))

    # determine setup class from given Python file
    setup_module = _import_from_file(setup_file)

    SetupClass = None
    for obj in vars(setup_module).values():
        if inspect.isclass(obj) and issubclass(obj, VerosSetup) and obj is not VerosSetup:
            if SetupClass is not None and SetupClass is not obj:
                raise RuntimeError("Veros setups can only define one VerosSetup class")

            SetupClass = obj

    from veros import logger

    target_version = getattr(setup_module, "__VEROS_VERSION__", None)
    if target_version and target_version != veros_version:
        logger.warning(
            f"This is Veros v{veros_version}, but the given setup was generated with v{target_version}. "
            "Consider switching to this version of Veros or updating your setup file.\n"
        )

    sim = SetupClass(*args, **kwargs)
    sim.setup()
    sim.run()


@click.command("veros-run")
@click.argument("SETUP_FILE", type=click.Path(readable=True, dir_okay=False, resolve_path=True))
@click.option(
    "-b",
    "--backend",
    default="numpy",
    type=click.Choice(BACKENDS),
    help="Backend to use for computations",
    show_default=True,
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(DEVICES),
    help="Hardware device to use (JAX backend only)",
    show_default=True,
)
@click.option(
    "-v",
    "--loglevel",
    default="info",
    type=click.Choice(LOGLEVELS),
    help="Log level used for output",
    show_default=True,
)
@click.option(
    "-s",
    "--override",
    nargs=2,
    multiple=True,
    metavar="SETTING VALUE",
    type=VerosSetting(),
    default=tuple(),
    help="Override model setting, may be specified multiple times",
)
@click.option(
    "-p",
    "--profile-mode",
    is_flag=True,
    default=False,
    type=click.BOOL,
    envvar="VEROS_PROFILE",
    help="Write a performance profile for debugging",
    show_default=True,
)
@click.option("--force-overwrite", is_flag=True, help="Silently overwrite existing outputs")
@click.option("--diskless-mode", is_flag=True, help="Supress all output to disk")
@click.option(
    "--float-type",
    default="float64",
    type=click.Choice(FLOAT_TYPES),
    help="Floating point precision to use",
    show_default=True,
)
@click.option(
    "-n", "--num-proc", nargs=2, default=[1, 1], type=click.INT, help="Number of processes in x and y dimension"
)
@functools.wraps(run)
def cli(setup_file, *args, **kwargs):
    if not setup_file.endswith(".py"):
        raise click.Abort(f"The given setup file {setup_file} does not appear to be a Python file.")

    return run(setup_file, *args, **kwargs)
