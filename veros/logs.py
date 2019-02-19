import logging


def setup_logging(loglevel="info", logfile=None):
    try: # python 2
        logging.basicConfig(logfile=logfile, filemode="w",
                            level=getattr(logging, loglevel.upper()),
                            format="%(message)s")
    except ValueError: # python 3
        logging.basicConfig(filename=logfile, filemode="w",
                            level=getattr(logging, loglevel.upper()),
                            format="%(message)s")
