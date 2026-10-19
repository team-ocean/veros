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

Available diagnostics
---------------------

Currently, the following diagnostics are implemented and added to
:obj:`VerosState.diagnostics`:

Snapshot
++++++++

.. autoclass:: veros.diagnostics.snapshot.Snapshot
   :members: name, output_variables, sampling_frequency, output_frequency, output_path, tendency_variables

Automatic snapshots select completed state variables from ``taup1`` before the
time indices rotate, so these states match the NetCDF ``Time`` coordinate.
Manual calls to ``Snapshot.output(state)`` select ``tau`` by default, as appropriate
after setup or after a timestep has returned.

Raw tendencies use their own storage conventions. Variables listed in
``snapshot.tendency_variables`` retain ``tau`` during automatic output; add custom
tendencies stored at ``tau`` to this set. The current streamfunction solve stores
``dpsi`` at ``taup1``, so it is intentionally absent from the set. These outputs are
raw stage tendencies, not states at ``Time`` or necessarily the realized rate over
an output interval.

Custom Snapshot subclasses overriding ``output`` must accept the optional keyword
``time_index`` and forward it to ``super().output``. Other diagnostic classes keep
their existing ``output(state)`` interface and sampling behavior.

Averages
++++++++

.. autoclass:: veros.diagnostics.averages.Averages
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

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
