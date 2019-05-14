import sys

from loguru import logger

try:
    import tqdm
except ImportError:
    has_tqdm = False
else:
    has_tqdm = True

import .time


def _noop(*args, **kwargs):
    pass


class DummyPBar:
    __enter__ = _noop
    __exit__ = _noop
    set_postfix = _noop
    
    def update(self, amount, *args, **kwargs):
        logger.info('Current time: {}', amount)


def get_progress_bar(vs):
    use_tqdm = sys.stdout.isatty() and rst.proc_num == 1

        if use_pbar:
            pbar = tqdm.tqdm(
                total=total_runlen,
                unit='(model {})'.format(time_unit.rstrip('s')),
                desc='Integrating',
                file=sys.stdout,
                bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} %s [{elapsed}<{remaining}, {rate_fmt}{postfix}]' % time_unit,
                disable=not use_pbar,
            )