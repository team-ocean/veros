from __future__ import absolute_import

import time


class Timer:
    def __init__(self, name):
        self.name = name
        self.total_time = 0
        self.last_time = 0

        try:
            import bohrium as bh
            flush = bh.flush
        except ImportError:
            def flush():
                pass

        self._flush = flush

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self._flush()
        self.last_time = time.time() - self.start_time
        self.total_time += self.last_time

    def printTime(self):
        print("[{}]: {}s".format(self.name, self.getTime()))

    def getTime(self):
        return self.total_time

    def getLastTime(self):
        return self.last_time
