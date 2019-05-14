import sys
import functools

import tqdm
from loguru import logger


def setup_logging(loglevel="info", logfile=None):
    from . import runtime_state

    if runtime_state.proc_rank != 0:
        logger.disable("veros")
        return

    kwargs = {}
    if sys.stdout.isatty():
        kwargs.update(
            sink=functools.partial(tqdm.tqdm.write, end=""),
            colorize=True
        )
    else:
        kwargs.update(
            sink=sys.stdout,
            colorize=False
        )

    config = {
        "handlers": [
            dict(
                level=loglevel.upper(),
                format="<level>{message}</level>",
                **kwargs
            )
        ]
    }

    if logfile is not None:
        config["handlers"].append(
            dict(sink=logfile, serialize=True)
        )

    logger.configure(**config)
    logger.enable("veros")
