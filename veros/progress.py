import sys
from time import perf_counter

from veros import logger, time, logs, runtime_state as rst

BAR_FORMAT = (
    " Current iteration: {iteration:<5} ({time:.2f}/{total:.2f}{unit} | {percentage:>4.1f}% | "
    "{rate:.2f}{rate_unit} | {eta:.1f}{eta_unit} left)"
)


class LoggingProgressBar:
    """A simple progress report to logger.info"""

    def __init__(self, total, start_time=0, start_iteration=0, burnin=0, fancy=False):
        self._start_time = start_time
        self._start_iteration = start_iteration
        self._total = total
        self._burnin = burnin
        self._start = None
        self._fancy = fancy

        _, self._time_unit = time.format_time(total)

    def __enter__(self):
        if self._fancy:
            logs.update_logging(stream_sink=self.write)

        self._iteration = self._start_iteration
        self._time = self._start_time
        self.flush()

        if not self._burnin:
            self._start = perf_counter()

        return self

    def write(self, msg):
        if msg.startswith("\r"):
            msg = msg.rstrip("\n")
        else:
            msg = "\n" + msg

        sys.stdout.write(msg)

    def __exit__(self, *args, **kwargs):
        if self._fancy:
            logs.update_logging()

    def advance_time(self, amount):
        self._iteration += 1
        self._time += amount
        self.flush()

        if self._iteration - self._start_iteration == self._burnin:
            self._start = perf_counter()
            self._start_time = self._time

    def flush(self):
        report_time = time.convert_time(self._time, "seconds", self._time_unit)
        total_time = time.convert_time(self._total, "seconds", self._time_unit)

        if self._start is not None:
            rate_in_seconds = (perf_counter() - self._start) / (self._time - self._start_time)
        else:
            rate_in_seconds = 0
        rate_in_seconds_per_year = rate_in_seconds / time.convert_time(1, "seconds", "years")

        rate, rate_unit = time.format_time(rate_in_seconds_per_year)
        eta, eta_unit = time.format_time((self._total - self._time) * rate_in_seconds)

        if self._start_time < self._total:
            percentage = 100 * (self._time - self._start_time) / (self._total - self._start_time)
        else:
            percentage = 100

        msg = BAR_FORMAT
        if self._fancy:
            msg = "\r" + msg

        logger.info(
            msg,
            time=report_time,
            total=total_time,
            unit=self._time_unit[0],
            percentage=percentage,
            iteration=self._iteration,
            rate=rate,
            rate_unit=f"{rate_unit[0]}/(model year)",
            eta=eta,
            eta_unit=eta_unit[0],
        )


def get_progress_bar(state, fancy=None, burnin=0):
    if fancy is None:
        fancy = sys.stdout.isatty() and rst.proc_num == 1

    kwargs = dict(
        total=state.settings.runlen + float(state.variables.time),
        start_time=float(state.variables.time),
        start_iteration=int(state.variables.itt),
        burnin=burnin,
        fancy=fancy,
    )

    pbar = LoggingProgressBar(**kwargs)
    return pbar
