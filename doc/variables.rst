Model variables
------------------

:fa:`clock-o`: Variable is time-dependent

:fa:`download`: Variable is written in snapshot output by default

:fa:`bar-chart`: Variable is written to average output by default

.. exec::
  from climate.pyom.variables import MAIN_VARIABLES, CONDITIONAL_VARIABLES
  for condition, vardict in [(None, MAIN_VARIABLES)] + CONDITIONAL_VARIABLES.items():
      if condition:
          print(condition)
      print("")
      headings = ["Flags", "Variable", "Units", "Dimensions", "Description", "Type"]
      print(".. csv-table:: Conditional variables")
      print("   :header: {}".format(" ,".join(headings)))
      print("   :widths: auto")
      print("")
      for key, var in vardict.items():
          units = ":math:`{}`".format(var.units) if var.units else ""
          flags = ""
          if var.time_dependent:
              flags += ":fa:`clock-o` "
          if var.output:
              flags += ":fa:`download` "
          if var.average:
              flags += ":fa:`bar-chart` "
          print('   "{}", "{}", "{}", "{}", "{}", "``{}``"'.format(flags, key, units, ", ".join(var.dims), var.long_description, var.dtype))
      print("")
