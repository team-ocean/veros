import timeit


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
        self.start_time = timeit.default_timer()

    def __exit__(self, type, value, traceback):
        self._flush()
        self.last_time = timeit.default_timer() - self.start_time
        self.total_time += self.last_time

    def print_time(self):
        print('[{}]: {}s'.format(self.name, self.get_time()))

    def get_time(self):
        return self.total_time

    def get_last_time(self):
        return self.last_time
