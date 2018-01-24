import functools

import click

from veros.backend import BACKENDS
from veros.settings import SETTINGS


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
                  help="Backend to use for computations (default: numpy)")
    @click.option("-v", "--loglevel", default="info",
                  type=click.Choice(["debug", "info", "warning", "error", "critical"]),
                  help="Log level used for output (default: info)")
    @click.option("-l", "--logfile", default=None,
                  help="Log file to write to (default: write to stdout)")
    @click.option("-s", "--override", nargs=2, multiple=True, metavar="SETTING VALUE",
                  type=VerosSetting(), default=tuple(),
                  help="Override default setting, may be specified multiple times")
    @click.option("-p", "--profile", is_flag=True, default=False, type=click.BOOL,
                  help="Write a performance profile for debugging (default: false)")
    @functools.wraps(run)
    def wrapped(*args, **kwargs):
        kwargs["override"] = dict(kwargs["override"])
        run(*args, **kwargs)

    return wrapped
