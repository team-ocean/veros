import logging

from . import veros_method, diagnostics_tools

@veros_method
def init_diagnostics(veros):
    """
    initialize diagnostic routines
    """
    logging.info("Diagnostic setup:")
    for name, diag in veros.diagnostics.items():
        if diag.is_active():
            try:
                diag_routine = getattr(diagnostics_tools, name)
            except AttributeError:
                raise AttributeError("unknown diagnostic {}".format(name))
            if diag.sampling_frequency:
                logging.info(" running diagnostic '{}' every {} seconds / {} time steps"
                            .format(name, diag.sampling_frequency, diag.sampling_frequency / veros.dt_tracer))
            if diag.output_frequency:
                logging.info(" writing output for diagnostic '{}' every {} seconds / {} time steps"
                            .format(name, diag.output_frequency, diag.output_frequency / veros.dt_tracer))
            diag_routine.initialize(veros)

@veros_method
def read_restart(veros):
    pass

@veros_method
def diagnose(veros):
    """
    call diagnostic routines
    """
    time = veros.itt * veros.dt_tracer
    for name, diag in veros.diagnostics.items():
        try:
            diag_routine = getattr(diagnostics_tools, name)
        except AttributeError:
            raise AttributeError("unknown diagnostic '{}'".format(name))
        if diag.sampling_frequency and time % diag.sampling_frequency < veros.dt_tracer:
            diag_routine.diagnose(veros)
        if diag.output_frequency and time % diag.output_frequency < veros.dt_tracer:
            diag_routine.output(veros)

@veros_method
def sanity_check(veros):
    if np.any(~np.isfinite(veros.u)):
        raise RuntimeError("solver diverged at iteration {}".format(veros.itt))

@veros_method
def panic_output(veros):
    logging.error("Writing snapshot before panic shutdown")
    if not veros.diagnostics["snapshot"].is_active():
        diagnostics_tools.snapshot.initialize(veros)
    diagnostics_tools.snapshot.diagnose(veros)
    diagnostics_tools.snapshot.output(veros)
