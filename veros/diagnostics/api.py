from veros import logger, time


def create_default_diagnostics(state):
    # do not import these at module level to make sure core import is deferred
    from veros.diagnostics.averages import Averages
    from veros.diagnostics.cfl_monitor import CFLMonitor
    from veros.diagnostics.energy import Energy
    from veros.diagnostics.overturning import Overturning
    from veros.diagnostics.snapshot import Snapshot
    from veros.diagnostics.tracer_monitor import TracerMonitor

    return {Diag.name: Diag(state) for Diag in (Averages, CFLMonitor, Energy, Overturning, Snapshot, TracerMonitor)}


def initialize(state):
    for name, diagnostic in state.diagnostics.items():
        diagnostic.initialize(state)

        if diagnostic.sampling_frequency:
            t, unit = time.format_time(diagnostic.sampling_frequency)
            logger.info(f' Running diagnostic "{name}" every {t:.1f} {unit}')

        if diagnostic.output_frequency:
            t, unit = time.format_time(diagnostic.output_frequency)
            logger.info(f' Writing output for diagnostic "{name}" every {t:.1f} {unit}')


def diagnose(state):
    vs = state.variables
    settings = state.settings

    for diagnostic in state.diagnostics.values():
        if diagnostic.sampling_frequency and vs.time % diagnostic.sampling_frequency < settings.dt_tracer:
            diagnostic.diagnose(state)


def output(state):
    vs = state.variables
    settings = state.settings

    for diagnostic in state.diagnostics.values():
        if diagnostic.output_frequency and vs.time % diagnostic.output_frequency < settings.dt_tracer:
            diagnostic.output(state)
