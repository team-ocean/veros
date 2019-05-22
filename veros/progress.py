import sys
import functools
from time import perf_counter

from loguru import logger

try:
    import tqdm
except ImportError:
    has_tqdm = False
else:
    has_tqdm = True

from . import time, logs, runtime_settings as rs, runtime_state as rst


class LoggingProgressBar:
    """A simple progress report to logger.info

    Serves as a fallback where TQDM is not available or not feasible (writing to a file,
    in multiprocessing contexts).
    """

    def __init__(self, total, start_time=0, start_iteration=0, time_unit='seconds'):
        self._start_time = start_time
        self._start_iteration = start_iteration
        self._total = total

        _, self._time_unit = time.format_time(total)

    def __enter__(self):
        self._start = perf_counter()
        self._iteration = self._start_iteration
        self._time = self._start_time

    def __exit__(self, *args, **kwargs):
        pass

    def advance_time(self, amount, *args, **kwargs):
        self._iteration += 1
        self._time += amount
        report_time = time.convert_time(self._time, 'seconds', self._time_unit)
        total_time = time.convert_time(self._total, 'seconds', self._time_unit)

        rate_in_seconds = (perf_counter() - self._start) / (self._time - self._start_time)
        rate_in_seconds_per_year = rate_in_seconds / time.convert_time(1, 'seconds', 'years')

        rate, rate_unit = time.format_time(rate_in_seconds_per_year)
        eta, eta_unit = time.format_time((self._total - self._time) * rate_in_seconds)
        percentage = 100 * (self._time - self._start_time) / (self._total - self._start_time)

        logger.info(
            ' Current iteration: {it:<5} ({time:.2f}/{total:.2f}{unit} | {percentage:<4.1f}% | '
            '{rate:.2f}{rate_unit} | {eta:.1f}{eta_unit} left)',
            time=report_time,
            total=total_time,
            unit=self._time_unit[0],
            percentage=percentage,
            it=self._iteration,
            rate=rate,
            rate_unit='{}/(model year)'.format(rate_unit[0]),
            eta=eta,
            eta_unit=eta_unit[0],
        )


class FancyProgressBar:
    """A fancy progress bar based on TQDM that stays at the bottom of the terminal."""

    def __init__(self, total, start_time=0, start_iteration=0, time_unit='seconds'):
        self._start_time = start_time
        self._start_iteration = start_iteration
        self._total = total

        total_runlen, time_unit = time.format_time(total)
        self._time_unit = time_unit

        bar_format = (
            '{{l_bar}}{{bar}}| {{n:.2f}}/{{total:.2f}}{time_unit}'
            ' [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]'.format(time_unit=time_unit[0])
        )
        self._pbar = tqdm.tqdm(
            total=total_runlen,
            unit='(model {})'.format(time_unit.rstrip('s')),
            desc='Integrating',
            file=sys.stdout,
            bar_format=bar_format,
            ncols=120,
        )

    def __enter__(self, *args, **kwargs):
        self._iteration = self._start_iteration
        self._time = self._start_time
        logs.setup_logging(loglevel=rs.loglevel, stream_sink=functools.partial(self._pbar.write, end=''))
        return self._pbar.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        logs.setup_logging(loglevel=rs.loglevel)
        return self._pbar.__exit__(*args, **kwargs)

    def advance_time(self, amount):
        amount_in_total_unit = min(
            time.convert_time(amount, 'seconds', self._time_unit),
            time.convert_time(self._total - self._time - 1e-8, 'seconds', self._time_unit)
        )
        self._pbar.update(amount_in_total_unit)

        self._iteration += 1
        self._time += amount

        total_time = time.convert_time(self._time, 'seconds', self._time_unit)
        self._pbar.set_postfix({
            'time': '{time:.2f}{unit}'.format(time=total_time, unit=self._time_unit[0]),
            'iteration': str(self._iteration),
        })


def get_progress_bar(vs, use_tqdm=None):
    if use_tqdm is None:
        use_tqdm = sys.stdout.isatty() and rst.proc_num == 1 and has_tqdm

    if use_tqdm and not has_tqdm:
        raise RuntimeError('tqdm failed to import. Try `pip install tqdm` or set use_tqdm=False.')

    kwargs = dict(
        total=vs.runlen + vs.time,
        start_time=vs.time,
        start_iteration=vs.itt
    )

    if use_tqdm:
        pbar = FancyProgressBar(**kwargs)
    else:
        pbar = LoggingProgressBar(**kwargs)

    return pbar
