Diagnostics
===========

Base class
----------

.. autoclass:: climate.pyom.settings.Diagnostic

Available diagnostics
---------------------
.. exec::
  from climate.pyom.settings import DIAGNOSTICS_SETTINGS
  for key, var in DIAGNOSTICS_SETTINGS.items():
      print(".. py:data:: {}".format(key))
      print("")
      print("  :var sampling_frequency: {}".format(var.sampling_frequency))
      print("  :var output_frequency: {}".format(var.output_frequency))
      print("  :var outfile: {}".format(var.outfile))
      print("")
      print("  {}".format(var.description))
      print("")
