import signal
import contextlib

from loguru import logger


@contextlib.contextmanager
def signals_to_exception(signals=(signal.SIGINT, signal.SIGTERM)):
    """Context manager that makes sure that converts system signals to exceptions.

    This allows for a graceful exit after receiving SIGTERM (e.g. through
    `kill` on UNIX systems).

    Example:
       >>> with signals_to_exception():
       >>>     try:
       >>>         # do something
       >>>     except SystemExit:
       >>>         # graceful exit even upon receiving interrupt signal
    """
    def signal_to_exception(sig, frame):
        logger.critical('Received interrupt signal {}', sig)
        raise SystemExit('Aborted')

    old_signals = {}
    for s in signals:
        # override signals with our handler
        old_signals[s] = signal.getsignal(s)
        signal.signal(s, signal_to_exception)

    try:
        yield

    finally:
        # re-attach old signals
        for s in signals:
            signal.signal(s, old_signals[s])
