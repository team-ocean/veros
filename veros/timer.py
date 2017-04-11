from __future__ import absolute_import

import numpy as np

try:
    import bohrium as bh
    flush = bh.flush
except ImportError:
    flush = lambda: None

import time

class Timer:
    def __init__(self, name):
        self.name = name
        self.starts = []
        self.ends = []

    def __enter__(self):
        self.starts.append(time.time())

    def __exit__(self, type, value, traceback):
        flush()
        self.ends.append(time.time())

    def printTime(self):
        self._check_if_active()
        totalTime = sum([self.ends[i] - self.starts[i] for i in xrange(len(self.starts))])
        print "[%s]:" % self.name, totalTime, "s"

    def getTime(self):
        self._check_if_active()
        totalTime = sum([self.ends[i] - self.starts[i] for i in xrange(len(self.starts))])
        return totalTime

    def getLastTime(self):
        self._check_if_active()
        return self.ends[-1] - self.starts[-1]

    def _check_if_active(self):
        if len(self.ends) != len(self.starts):
            raise RuntimeError("must be called after Timer context ends")
