.. _diagnostics:

Diagnostics
===========

Diagnostics are separate objects (instances of subclasses of :class:`VerosDiagnostic`)
responsible for handling I/O, restart mechanics, and monitoring of the numerical
solution. All available diagnostics are instantiated and added to a dictionary
attribute :attr:`VerosState.diagnostics` (with a key determined by their `name` attribute).
Options for diagnostics may be set during the :meth:`VerosSetup.set_diagnostics` method:

::

   class MyModelSetup(VerosSetup):
       ...
       def set_diagnostics(self, state):
           diagnostics = state.diagnostics
           diagnostics['averages'].output_variables = ['psi','u','v']
           diagnostics['averages'].sampling_frequency = 3600.
           diagnostics['snapshot'].output_variables += ['du']

Base class
----------

This class implements some common logic for all diagnostics. This makes it easy
to write your own diagnostics: Just derive from this class, and implement the
virtual functions.

.. autoclass:: veros.diagnostics.base.VerosDiagnostic
   :members: name, initialize, diagnose, output

Automatic timestepping follows PyOM's batch-driver convention. After computing a
step, both ``diagnose(state)`` and ``output(state)`` run before index rotation and
clock advancement: ``tau`` and the clock describe the input state, while ``taup1``
contains the completed state. This lets energy diagnostics calculate changes over
the step and gives averages and overturning access to the input state. Only after
all diagnostics finish does the driver rotate the indices and advance ``itt`` and
``time``. The ``after_timestep`` hook runs before diagnostics, with the same clock
and indices.

Available diagnostics
---------------------

Currently, the following diagnostics are implemented and added to
:obj:`VerosState.diagnostics`:

Snapshot
++++++++

.. autoclass:: veros.diagnostics.snapshot.Snapshot
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

Snapshots select ``tau`` for every variable with a model timestep dimension.
Automatic snapshots write the input state with its matching NetCDF ``Time``
coordinate. A run with timesteps samples time zero when output is due; a
zero-length run produces no automatic snapshots. For example, a 60-second run
with 10-second steps and a 20-second output interval writes snapshots at 0, 20,
and 40 seconds. Finishing the run does not automatically write the final state.
An explicit ``Snapshot.output(state)`` after the run writes the completed state
at 60 seconds. Manual snapshots after setup or inside ``after_timestep`` also
select ``tau``, with a clock that describes that state.

Raw tendencies follow their solver storage conventions, without per-variable slot
adjustments. During automatic output, the latest ``dtemp``, ``dsalt``, and
``dpsin`` are at ``tau``, while the latest ``dpsi`` is at ``taup1``. Snapshots select
``tau`` for all of them, so ``dpsi`` retains the previous solve's value. After a
step returns, rotation has moved these latest standard tendencies to ``taum1``
and the latest ``dpsi`` to ``tau``; manual snapshots still select ``tau``. These
raw tendencies are not necessarily instantaneous rates at ``Time`` or realized
rates over an output interval.

Averages
++++++++

.. autoclass:: veros.diagnostics.averages.Averages
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

Sampling and output use the starting clock, including time zero. If both are due,
the current sample is included before output resets the accumulated sum. For the
example above, sampling every step gives an initial average at 0 seconds, then
averages of the 10/20-second and 30/40-second samples at 20 and 40 seconds. The
50-second sample remains accumulated for the next output or restart.

CFL monitor
+++++++++++

.. autoclass:: veros.diagnostics.cfl_monitor.CFLMonitor
   :members: name, sampling_frequency, output_frequency

Tracer monitor
++++++++++++++

.. autoclass:: veros.diagnostics.tracer_monitor.TracerMonitor
   :members: name, sampling_frequency, output_frequency

Energy
++++++

.. autoclass:: veros.diagnostics.energy.Energy
   :members: name, sampling_frequency, output_frequency, output_path

Overturning
+++++++++++

.. autoclass:: veros.diagnostics.overturning.Overturning
   :members: name, p_ref, sampling_frequency, output_frequency, output_path
