Diagnostics
===========

Base class
----------

.. autoclass:: veros.settings.Diagnostic

Available diagnostics
---------------------
.. exec::
  from veros.settings import DIAGNOSTICS_SETTINGS
  for key, var in DIAGNOSTICS_SETTINGS.items():
      print(".. py:data:: {}".format(key))
      print("")
      print("  :var sampling_frequency: {}".format(var.sampling_frequency))
      print("  :var output_frequency: {}".format(var.output_frequency))
      print("  :var outfile: {}".format(var.outfile))
      print("")
      print("  {}".format(var.description))
      print("")
