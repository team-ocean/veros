from . import averages, cfl_monitor, energy, overturning, snapshot, tracer_monitor, io_tools

diagnostics = {diag.name: diag for diag in (averages.Averages, cfl_monitor.CFLMonitor,
                                            energy.Energy, overturning.Overturning,
                                            snapshot.Snapshot, tracer_monitor.TracerMonitor)}
