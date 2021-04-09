from loguru import logger
import numpy as np

from . import averages, cfl_monitor, energy, overturning, snapshot, tracer_monitor, io_tools
from .. import time
from .io_tools import hdf5 as h5tools


def create_default_diagnostics():
    return {Diag.name: Diag() for Diag in (averages.Averages, cfl_monitor.CFLMonitor,
                                           energy.Energy, overturning.Overturning,
                                           snapshot.Snapshot, tracer_monitor.TracerMonitor)}


def sanity_check(vs):
    from ..distributed import global_and
    return global_and(np.all(np.isfinite(vs.u)))


def read_restart(vs):
    if not vs.restart_input_filename:
        return
    if vs.force_overwrite:
        raise RuntimeError('To prevent data loss, force_overwrite cannot be used in restart runs')
    logger.info('Reading restarts')
    for diagnostic in vs.diagnostics.values():
        diagnostic.read_restart(vs, vs.restart_input_filename.format(**vars(vs)))


def write_restart(vs, force=False):
    if vs.diskless_mode:
        return
    if not vs.restart_output_filename:
        return
    if force or vs.restart_frequency and vs.time % vs.restart_frequency < vs.dt_tracer:
        output_filename = vs.restart_output_filename.format(**vars(vs))
        logger.info('Writing restart file {}...', output_filename)
        with h5tools.threaded_io(vs, output_filename, 'w') as outfile:
            for diagnostic in vs.diagnostics.values():
                diagnostic.write_restart(vs, outfile)


def initialize(vs):
    for name, diagnostic in vs.diagnostics.items():
        diagnostic.initialize(vs)
        if diagnostic.sampling_frequency:
            logger.info(' Running diagnostic "{0}" every {1[0]:.1f} {1[1]}'
                         .format(name, time.format_time(diagnostic.sampling_frequency)))
        if diagnostic.output_frequency:
            logger.info(' Writing output for diagnostic "{0}" every {1[0]:.1f} {1[1]}'
                         .format(name, time.format_time(diagnostic.output_frequency)))


def diagnose(vs):
    for diagnostic in vs.diagnostics.values():
        if diagnostic.sampling_frequency and vs.time % diagnostic.sampling_frequency < vs.dt_tracer:
            diagnostic.diagnose(vs)


def output(vs):
    for diagnostic in vs.diagnostics.values():
        if diagnostic.output_frequency and vs.time % diagnostic.output_frequency < vs.dt_tracer:
            diagnostic.output(vs)
