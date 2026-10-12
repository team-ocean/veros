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
   :members: name, initialize, diagnose, output, finalize, write_finalized, supports_compilation, observes_model

Built-in diagnostics separate numerical calculation from output. Calling
:meth:`VerosDiagnostic.finalize` returns a dictionary of output values and updates
the diagnostic's accumulation window or history. Snapshot values select the
current model time step. Calling :meth:`VerosDiagnostic.write_finalized` writes
those values without resetting accumulators and accepts an optional ``time_value``
in seconds for delayed output. The existing :meth:`VerosDiagnostic.output` method
calls both methods in sequence.

Custom diagnostics can continue implementing ``initialize``, ``diagnose``, and
``output``. Set ``supports_compilation`` only when their numerical methods support
JAX tracing, and implement ``finalize`` to return output values without writing
files or logs. The ``observes_model`` attribute declares that diagnostic methods
leave model variables and settings unchanged.

Available diagnostics
---------------------

Currently, the following diagnostics are implemented and added to
:obj:`VerosState.diagnostics`:

Snapshot
++++++++

.. autoclass:: veros.diagnostics.snapshot.Snapshot
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

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
