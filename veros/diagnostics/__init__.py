import logging

from . import averages, cfl_monitor, energy, overturning, snapshot, tracer_monitor, io_tools
from .. import time, veros_method
from .io_tools import hdf5 as h5tools


@veros_method
def create_diagnostics(veros):
    return {Diag.name: Diag(veros) for Diag in (averages.Averages, cfl_monitor.CFLMonitor,
                                                energy.Energy, overturning.Overturning,
                                                snapshot.Snapshot, tracer_monitor.TracerMonitor)}

@veros_method
def sanity_check(veros):
    return np.all(np.isfinite(veros.u))

@veros_method
def read_restart(veros):
    if not veros.restart_input_filename:
        return
    logging.info("Reading restarts")
    for diagnostic in veros.diagnostics.values():
        diagnostic.read_restart(veros)

@veros_method
def write_restart(veros, force=False):
    if veros.diskless_mode:
        return
    t = time.current_time(veros, "seconds")
    if force or veros.restart_frequency and t % veros.restart_frequency < veros.dt_tracer:
        with h5tools.threaded_io(veros, veros.restart_output_filename.format(**vars(veros)), "w") as outfile:
            for diagnostic in veros.diagnostics.values():
                diagnostic.write_restart(veros, outfile)

@veros_method
def initialize(veros):
    for name, diagnostic in veros.diagnostics.items():
        diagnostic.initialize(veros)
        if diagnostic.sampling_frequency:
            logging.info(" running diagnostic '{0}' every {1[0]:.1f} {1[1]} / {2:.1f} time steps"
                         .format(name, time.format_time(veros, diagnostic.sampling_frequency),
                                 diagnostic.sampling_frequency / veros.dt_tracer))
        if diagnostic.output_frequency:
            logging.info(" writing output for diagnostic '{0}' every {1[0]:.1f} {1[1]} / {2:.1f} time steps"
                         .format(name, time.format_time(veros, diagnostic.output_frequency),
                                 diagnostic.output_frequency / veros.dt_tracer))

@veros_method
def diagnose(veros):
    t = time.current_time(veros, "seconds")
    for diagnostic in veros.diagnostics.values():
        if diagnostic.sampling_frequency and t % diagnostic.sampling_frequency < veros.dt_tracer:
            diagnostic.diagnose(veros)

@veros_method
def output(veros):
    t = time.current_time(veros, "seconds")
    for diagnostic in veros.diagnostics.values():
        if diagnostic.output_frequency and t % diagnostic.output_frequency < veros.dt_tracer:
            diagnostic.output(veros)

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
