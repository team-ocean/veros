import timeit
import threading

timer_context = threading.local()
timer_context.active = True


class Timer:
    def __init__(self):
        self.total_time = 0
        self.last_time = 0

    def __enter__(self):
        self.start_time = timeit.default_timer()

    def __exit__(self, *args, **kwargs):
        self.last_time = timeit.default_timer() - self.start_time

        if timer_context.active:
            self.total_time += self.last_time
