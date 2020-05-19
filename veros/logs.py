import sys
import warnings

from loguru import logger

# register custom loglevel
logger.level('DIAGNOSTIC', no=45)


def setup_logging(loglevel='info', stream_sink=sys.stdout):
    from . import runtime_state

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
    logger.level('SUCCESS', color='<dim><green>')
    logger.level('WARNING', color='<yellow>')
    logger.level('ERROR', color='<bold><red>')
    logger.level('DIAGNOSTIC', color='<bold><yellow>')
    logger.level('CRITICAL', color='<bold><red><WHITE>')

    config = {
        'handlers': [
            dict(
                sink=stream_sink,
                level=loglevel.upper(),
                format='<level>{message}</level>',
                filter=lambda record: runtime_state.proc_rank == 0,
                **kwargs
            )
        ]
    }

    def diagnostic(_, message, *args, **kwargs):
        logger.opt(depth=1).log('DIAGNOSTIC', message, *args, **kwargs)

    logger.__class__.diagnostic = diagnostic

    def showwarning(message, cls, source, lineno, *args):
        logger.warning(
            '{warning}: {message} ({source}:{lineno})',
            message=message,
            warning=cls.__name__,
            source=source,
            lineno=lineno
        )

    warnings.showwarning = showwarning

    if runtime_state.proc_rank == 0:
        logger.enable('veros')
    else:
        logger.disable('veros')

    return logger.configure(**config)
