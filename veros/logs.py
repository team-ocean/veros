import sys
import warnings

from loguru import logger


def setup_logging(loglevel='info', stream_sink=sys.stdout):
    from . import runtime_state

    if runtime_state.proc_rank != 0:
        logger.disable('veros')
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
        'handlers': [
            dict(
                sink=stream_sink,
                level=loglevel.upper(),
                format='<level>{message}</level>',
                **kwargs
            )
        ]
    }

    def showwarning(message, cls, source, lineno, *args):
        logger.warning(
            '{warning}: {message} ({source}:{lineno})',
            message=message,
            warning=cls.__name__,
            source=source,
            lineno=lineno
        )

    warnings.showwarning = showwarning

    logger.enable('veros')
    return logger.configure(**config)
