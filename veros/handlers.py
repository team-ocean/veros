import signal
import contextlib


@contextlib.contextmanager
def signals_to_exception(signals=(signal.SIGINT, signal.SIGTERM)):
    """Context manager that makes sure that converts system signals to exceptions.

    This allows e.g. for a graceful exit after receiving SIGTERM (e.g. through
    `kill` on UNIX systems).

    Example:
       >>> with signals_to_exception():
       >>>     try:
       >>>         # do something
       >>>     except SystemExit:
       >>>         # graceful exit even upon receiving interrupt signal
    """
    def signal_to_exception(sig, frame):
        raise SystemExit("received interrupt signal {}".format(sig))
    old_signals = {}
    for s in signals:
        old_signals[s] = signal.getsignal(s)
        signal.signal(s, signal_to_exception)
    try:
        yield
    finally:
        for s in signals:
            signal.signal(s, old_signals[s])
