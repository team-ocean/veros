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

During automatic timestepping, ``diagnose(state)`` runs before time indices rotate:
``tau`` contains the input state and ``taup1`` the completed state. This lets
diagnostics such as energy calculate changes over the step, and preserves the
input-state sampling used by averages and overturning. ``output(state)`` runs
after rotation, when ``tau`` contains the completed state. The model clock has
already advanced in both stages. Custom diagnostics needing both states should
perform those calculations in ``diagnose``.

Available diagnostics
---------------------

Currently, the following diagnostics are implemented and added to
:obj:`VerosState.diagnostics`:

Snapshot
++++++++

.. autoclass:: veros.diagnostics.snapshot.Snapshot
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

Snapshots select ``tau`` for every variable with a model timestep dimension.
Automatic snapshots run after rotation, so completed state variables match the
NetCDF ``Time`` coordinate. Manual calls to ``Snapshot.output(state)`` use the same
selection, as appropriate after setup or after a timestep has returned. The
``after_timestep`` hook still runs before rotation; a manual snapshot inside that
hook selects the input state.

Raw tendencies follow their solver storage conventions, without per-variable slot
adjustments. For example, after rotation the latest ``dtemp``, ``dsalt``, and
``dpsin`` are at ``taum1``, while the latest ``dpsi`` is at ``tau``. Snapshots select
``tau`` for all of them, which can include older or initial slot contents. These
raw tendencies are not necessarily instantaneous rates at ``Time`` or realized
rates over an output interval.

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
