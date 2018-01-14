import argparse

from .backend import BACKENDS
from .settings import SETTINGS

def parse_command_line():
    parser = argparse.ArgumentParser(description="Veros command line interface")
    parser.add_argument("--backend", "-b", default="numpy", choices=BACKENDS,
                        help="Backend to use for computations. Defaults to 'numpy'.")
    parser.add_argument("--loglevel", "-v", default="info",
                        choices=("debug", "info", "warning", "error", "critical"),
                        help="Log level used for output. Defaults to 'info'.")
    parser.add_argument("--logfile", "-l", default=None,
                        help="Log file to write to. Writing to stdout if not set.")
    parser.add_argument("--profile", "-p", default=False, action="store_true",
                        help="Profile Veros using pyinstrument")
    parser.add_argument("--set", "-s", nargs=2, action="append", metavar=("SETTING", "VALUE"),
                        help="Override default setting. May be specified multiple times.")
    args, _ = parser.parse_known_args()
    return args

def set_commandline_settings(vs):
    for key, val in vs.command_line_settings:
        setting = SETTINGS[key]
        if setting.type is bool:
            val = str_to_bool(val)
        setattr(vs, key, setting.type(val))

def str_to_bool(string):
    return string.lower() in ("1", "true", "on")
