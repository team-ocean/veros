import time

class Timer:
    def __init__(self, name):
        self.name = name
        self.starts = []
        self.ends = []

    def __enter__(self):
        self.starts.append(time.time())

    def __exit__(self, type, value, traceback):
        self.ends.append(time.time())

    def printTime(self):
        totalTime = sum([self.ends[i] - self.starts[i] for i in xrange(len(self.starts))])
        print "[%s]:" % self.name, totalTime, "s"

    def getTime(self):
        totalTime = sum([self.ends[i] - self.starts[i] for i in xrange(len(self.starts))])
        return totalTime
