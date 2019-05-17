.. _variables:

Model variables
---------------

The variable meta-data (i.e., all instances of :class:`veros.variables.Variable`)
are available in a dictionary as the attribute :attr:`Veros.variables`. The actual
data arrays are added directly as attributes to :class:`Veros`. The following code
snippet (as commonly used in the :ref:`diagnostics`) illustrates this behavior:

::

   var_meta = {key: val for key, val in vs.variables.items() if val.time_dependent and val.output}
   var_data = {key: getattr(veros, key) for key in var_meta.keys()}

In this case, ``var_meta`` is a dictionary containing all metadata for variables that
are time dependent and should be added to the output, while ``var_data`` is a dictionary
with the same keys containing the corresponding data arrays.

Variable class
++++++++++++++

.. autoclass:: veros.variables.Variable

Available variables
+++++++++++++++++++

There are two kinds of variables in Veros. Main variables are always present in a
simulation, while conditional variables are only available if their respective
condition is :obj:`True` at the time of variable allocation.

.. _flag_legend:

Attributes:
  | :fa:`clock-o`: Time-dependent
  | :fa:`download`: Included in snapshot output by default
  | :fa:`repeat`: Written to restart files by default

.. exec::
  from veros.variables import MAIN_VARIABLES, CONDITIONAL_VARIABLES
  first_condition = True
  for condition, vardict in [(None, MAIN_VARIABLES)] + list(CONDITIONAL_VARIABLES.items()):
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
          if var.write_to_restart:
              flags += ":fa:`repeat` "
          print(".. py:attribute:: Veros.{}".format(key))
          print("")
          print("  :units: {}".format(var.units))
          print("  :dimensions: {}".format(", ".join(var.dims)))
          print("  :type: :py:class:`{}`".format(var.dtype or "float"))
          print("  :attributes: {}".format(flags))
          print("")
          print("  {}".format(var.long_description))
          print("")
