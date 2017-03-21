import logging

from . import pyom_method, diagnostics_tools

@pyom_method
def init_diagnostics(pyom):
    """
    initialize diagnostic routines
    """
    logging.info("Diagnostic setup:")
    for name, diag in pyom.diagnostics.items():
        if diag.is_active():
            try:
                diag_routine = getattr(diagnostics_tools, name)
            except AttributeError:
                raise AttributeError("unknown diagnostic {}".format(name))
            if diag.sampling_frequency:
                logging.info(" running diagnostic '{}' every {} seconds / {} time steps"
                            .format(name, diag.sampling_frequency, diag.sampling_frequency / pyom.dt_tracer))
            if diag.output_frequency:
                logging.info(" writing output for diagnostic '{}' every {} seconds / {} time steps"
                            .format(name, diag.output_frequency, diag.output_frequency / pyom.dt_tracer))
            diag_routine.initialize(pyom)

@pyom_method
def read_restart(pyom):
    pass

@pyom_method
def diagnose(pyom):
    """
    call diagnostic routines
    """
    time = pyom.itt * pyom.dt_tracer
    for name, diag in pyom.diagnostics.items():
        try:
            diag_routine = getattr(diagnostics_tools, name)
        except AttributeError:
            raise AttributeError("unknown diagnostic '{}'".format(name))
        if diag.sampling_frequency and time % diag.sampling_frequency < pyom.dt_tracer:
            diag_routine.diagnose(pyom)
        if diag.output_frequency and time % diag.output_frequency < pyom.dt_tracer:
            diag_routine.output(pyom)

@pyom_method
def sanity_check(pyom):
    if np.any(~np.isfinite(pyom.u)):
        raise RuntimeError("solver diverged at iteration {}".format(pyom.itt))

@pyom_method
def panic_output(pyom):
    logging.error("Writing snapshot before panic shutdown")
    if not pyom.diagnostics["snapshot"].is_active():
        diagnostics_tools.snapshot.initialize(pyom)
    diagnostics_tools.snapshot.diagnose(pyom)
    diagnostics_tools.snapshot.output(pyom)
