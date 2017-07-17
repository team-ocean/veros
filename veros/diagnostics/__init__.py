import logging

from . import averages, cfl_monitor, energy, overturning, snapshot, tracer_monitor, io_tools
from .. import time, veros_method
from .io_tools import hdf5 as h5tools


@veros_method
def create_diagnostics(vs):
    return {Diag.name: Diag(vs) for Diag in (averages.Averages, cfl_monitor.CFLMonitor,
                                                energy.Energy, overturning.Overturning,
                                                snapshot.Snapshot, tracer_monitor.TracerMonitor)}


@veros_method
def sanity_check(vs):
    return np.all(np.isfinite(vs.u))


@veros_method
def read_restart(vs):
    if not vs.restart_input_filename:
        return
    if vs.force_overwrite:
        raise RuntimeError("to prevent data loss, force_overwrite cannot be used in restart runs")
    logging.info("Reading restarts")
    for diagnostic in vs.diagnostics.values():
        diagnostic.read_restart(vs)


@veros_method
def write_restart(vs, force=False):
    if vs.diskless_mode:
        return
    if force or vs.restart_frequency and vs.time % vs.restart_frequency < vs.dt_tracer:
        with h5tools.threaded_io(vs, vs.restart_output_filename.format(**vars(vs)), "w") as outfile:
            for diagnostic in vs.diagnostics.values():
                diagnostic.write_restart(vs, outfile)


@veros_method
def initialize(vs):
    for name, diagnostic in vs.diagnostics.items():
        diagnostic.initialize(vs)
        if diagnostic.sampling_frequency:
            logging.info(" running diagnostic '{0}' every {1[0]:.1f} {1[1]}"
                         .format(name, time.format_time(vs, diagnostic.sampling_frequency)))
        if diagnostic.output_frequency:
            logging.info(" writing output for diagnostic '{0}' every {1[0]:.1f} {1[1]}"
                         .format(name, time.format_time(vs, diagnostic.output_frequency)))


@veros_method
def diagnose(vs):
    for diagnostic in vs.diagnostics.values():
        if diagnostic.sampling_frequency and vs.time % diagnostic.sampling_frequency < vs.dt_tracer:
            diagnostic.diagnose(vs)


@veros_method
def output(vs):
    for diagnostic in vs.diagnostics.values():
        if diagnostic.output_frequency and vs.time % diagnostic.output_frequency < vs.dt_tracer:
            diagnostic.output(vs)


def start_profiler():
    import pyinstrument
    profiler = pyinstrument.Profiler()
    profiler.start()
    return profiler


def stop_profiler(profiler):
    try:
        profiler.stop()
        with open("profile.html", "w") as f:
            f.write(profiler.output_html())
    except UnboundLocalError:  # profiler has not been started
        pass
