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

    logger.level('TRACE', color='<dim>')
    logger.level('DEBUG', color='<dim><cyan>')
    logger.level('INFO', color='')
    logger.level('WARNING', color='<yellow>')
    logger.level('ERROR', color='<bold><red>')
    logger.level('CRITICAL', color='<bold><red><WHITE>')
    logger.level('SUCCESS', color='<dim><green>')

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
