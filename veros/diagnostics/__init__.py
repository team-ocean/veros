from . import averages, cfl_monitor, energy, overturning, snapshot, tracer_monitor

diagnostics = {
    "averages": averages.Averages,
    "cfl_monitor": cfl_monitor.CFLMonitor,
    "energy": energy.Energy,
    "overturning": overturning.Overturning,
    "snapshot": snapshot.Snapshot,
    "tracer_monitor": tracer_monitor.TracerMonitor,
}
