import logging

from . import runtime_state


def setup_logging(loglevel="info", logfile=None):
    if runtime_state.proc_rank != 0:
        return

    try: # python 2
        logging.basicConfig(logfile=logfile, filemode="w",
                            level=getattr(logging, loglevel.upper()),
                            format="%(message)s")
    except ValueError: # python 3
        logging.basicConfig(filename=logfile, filemode="w",
                            level=getattr(logging, loglevel.upper()),
                            format="%(message)s")
