import sys

from loguru import logger


def setup_logging(loglevel="info", logfile=None):
    from . import runtime_state

    if runtime_state.proc_rank != 0:
        logger.disable("veros")
        return

    config = {
        "handlers": [
            dict(
                sink=sys.stderr,
                colorize=sys.stderr.isatty(),
                level=loglevel.upper(),
                format="<level>{message}</level>"
            )
        ]
    }

    if logfile is not None:
        config["handlers"].append(
            dict(sink=logfile, serialize=True)
        )

    logger.configure(**config)
    logger.enable("veros")
