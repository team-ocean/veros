import sys
import warnings

from loguru import logger

# register custom loglevel
logger.level('DIAGNOSTIC', no=45)


def setup_logging(loglevel='info', stream_sink=sys.stdout):
    from . import runtime_state, runtime_settings

    handler_conf = dict(
        sink=stream_sink,
        level=loglevel.upper(),
        colorize=sys.stdout.isatty(),
    )

    logger.level('TRACE', color='<dim>')
    logger.level('DEBUG', color='<dim><cyan>')
    logger.level('INFO', color='')
    logger.level('SUCCESS', color='<dim><green>')
    logger.level('WARNING', color='<yellow>')
    logger.level('ERROR', color='<bold><red>')
    logger.level('DIAGNOSTIC', color='<bold><yellow>')
    logger.level('CRITICAL', color='<bold><red><WHITE>')

    if runtime_settings.log_all_processes:
        handler_conf.update(
            format=f'{runtime_state.proc_rank} | <level>{{message}}</level>'
        )
    else:
        handler_conf.update(
            format='<level>{message}</level>',
            filter=lambda record: runtime_state.proc_rank == 0
        )

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

    veros_logger = logger.configure(handlers=[handler_conf])
    logger.enable('veros')

    return veros_logger
