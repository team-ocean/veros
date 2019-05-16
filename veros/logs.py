import sys

from loguru import logger


def setup_logging(loglevel="info", stream_sink=sys.stdout):
    from . import runtime_state

    if runtime_state.proc_rank != 0:
        logger.disable("veros")
        return

    kwargs = {}
    if sys.stdout.isatty():
        kwargs.update(
            colorize=True
        )
    else:
        kwargs.update(
            colorize=False
        )

    config = {
        "handlers": [
            dict(
                sink=stream_sink,
                level=loglevel.upper(),
                format="<level>{message}</level>",
                **kwargs
            )
        ]
    }

    logger.enable("veros")
    return logger.configure(**config)
