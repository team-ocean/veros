Model variables
---------------

Variable base class
+++++++++++++++++++

.. autoclass:: veros.variables.Variable

Available variables
+++++++++++++++++++

.. _flag_legend:

Attributes:
  | :fa:`clock-o`: Time-dependent
  | :fa:`download`: Included in snapshot output by default
  | :fa:`bar-chart`: Included in average output by default

.. exec::
  from veros.variables import MAIN_VARIABLES, CONDITIONAL_VARIABLES
  first_condition = True
  for condition, vardict in [(None, MAIN_VARIABLES)] + CONDITIONAL_VARIABLES.items():
      if condition:
          if first_condition:
              print("Conditional variables")
              print("=====================")
              first_condition = False
          print(condition)
          print(len(condition) * "#")
      else:
          print("Main variables")
          print("==============")
      print("")
      for key, var in vardict.items():
          flags = ""
          if var.time_dependent:
              flags += ":fa:`clock-o` "
          if var.output:
              flags += ":fa:`download` "
          if var.average:
              flags += ":fa:`bar-chart` "
          print(".. py:attribute:: PyOM.{}".format(key))
          print("")
          print("  :units: {}".format(var.units))
          print("  :dimensions: {}".format(", ".join(var.dims)))
          print("  :type: :py:class:`{}`".format(var.dtype))
          print("  :attributes: {}".format(flags))
          print("")
          print("  {}".format(var.long_description))
          print("")
