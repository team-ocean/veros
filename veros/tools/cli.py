import functools

import click

from veros.settings import SETTINGS

BACKENDS = ["numpy", "bohrium"]
LOGLEVELS = ["trace", "debug", "info", "warning", "error", "critical"]


class VerosSetting(click.ParamType):
    name = "setting"
    current_key = None

    def convert(self, value, param, ctx):
        assert param.nargs == 2

        if self.current_key is None:
            if value not in SETTINGS:
                self.fail("Unknown setting %s" % value)
            self.current_key = value
            return value

        assert self.current_key in SETTINGS
        setting = SETTINGS[self.current_key]
        self.current_key = None

        if setting.type is bool:
            return click.BOOL(value)

        return setting.type(value)


def cli(run):
    @click.command("veros-run")
    @click.option("-b", "--backend", default="numpy", type=click.Choice(BACKENDS),
                  help="Backend to use for computations (default: numpy)", envvar="VEROS_BACKEND")
    @click.option("-v", "--loglevel", default="info", type=click.Choice(LOGLEVELS),
                  help="Log level used for output (default: info)", envvar="VEROS_LOGLEVEL")
    @click.option("-l", "--logfile", default=None, envvar="VEROS_LOGFILE",
                  help="Log file to write to (default: write to stdout)")
    @click.option("-s", "--override", nargs=2, multiple=True, metavar="SETTING VALUE",
                  type=VerosSetting(), default=tuple(),
                  help="Override default setting, may be specified multiple times")
    @click.option("-p", "--profile-mode", is_flag=True, default=False, type=click.BOOL, envvar="VEROS_PROFILE",
                  help="Write a performance profile for debugging (default: false)")
    @click.option("-n", "--num-proc", nargs=2, default=[1, 1], type=click.INT,
                  help="Number of processes in x and y dimension (requires execution via mpirun)")
    @functools.wraps(run)
    def wrapped(*args, **kwargs):
        from veros import runtime_settings

        kwargs["override"] = dict(kwargs["override"])

        for setting in ("backend", "profile_mode", "num_proc", "loglevel", "logfile"):
            if setting not in kwargs:
                continue
            setattr(runtime_settings, setting, kwargs.pop(setting))

        run(*args, **kwargs)

    return wrapped
