.. _diagnostics:

Diagnostics
===========

Diagnostics are separate objects (instances of subclasses of :class:`VerosDiagnostic`)
responsible for handling I/O, restart mechanics, and monitoring of the numerical
solution. All available diagnostics are instantiated and added to a dictionary
attribute :attr:`Veros.diagnostics` (with a key determined by their `name` attribute).
Options for diagnostics may be set during the :meth:`Veros.set_diagnostics` method:

::

   class MyModelSetup(Veros):
       ...
       def set_diagnostics(self):
           self.diagnostics["averages"].output_variables = ["psi","u","v"]
           self.diagnostics["averages"].sampling_frequency = 3600.
           self.diagnostics["snapshot"].output_variables += ["du"]

Base class
----------

This class implements some common logic for all diagnostics. This makes it easy
to write your own diagnostics: Just derive from this class, and implement the
virtual functions.

.. autoclass:: veros.diagnostics.diagnostic.VerosDiagnostic
   :members: name, initialize, diagnose, output, read_restart, write_restart

Available diagnostics
---------------------

Currently, the following diagnostics are implemented and added to
:obj:`Veros.diagnostics`:

Snapshot
++++++++

.. autoclass:: veros.diagnostics.snapshot.Snapshot
   :members: name, output_variables, restart_variables, sampling_frequency, output_frequency, output_path

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
