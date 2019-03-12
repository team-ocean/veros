import logging


class MyLogger(logging.getLoggerClass()):
    TRACE = 5

    def __init__(self, name, level=logging.NOTSET):
        super(MyLogger, self).__init__(name, level)
        logging.addLevelName(MyLogger.TRACE, "TRACE")

    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(MyLogger.TRACE):
            self._log(MyLogger.TRACE, msg, args, **kwargs)


logging.setLoggerClass(MyLogger)


def setup_logging(loglevel="info", logfile=None):
    from . import runtime_state

    if runtime_state.proc_rank != 0:
        logging.basicConfig(level="ERROR")
        return

    try: # python 2
        logging.basicConfig(logfile=logfile, filemode="w",
                            level=loglevel.upper(),
                            format="%(message)s")
    except ValueError: # python 3
        logging.basicConfig(filename=logfile, filemode="w",
                            level=loglevel.upper(),
                            format="%(message)s")
