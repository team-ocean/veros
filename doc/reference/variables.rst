.. _variables:

Model variables
===============

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
--------------

.. autoclass:: veros.variables.Variable

Available variables
-------------------

There are two kinds of variables in Veros. Main variables are always present in a
simulation, while conditional variables are only available if their respective
condition is :obj:`True` at the time of variable allocation.

.. _flag_legend:

Attributes:
  | :fa:`clock-o`: Time-dependent
  | :fa:`question-circle`: Conditional
  | :fa:`repeat`: Written to restart files by default

.. exec::
  import inspect
  from veros.variables import VARIABLES

  seen = set()

  for key, var in VARIABLES.items():
      is_conditional = callable(var.active)

      flags = ""
      if var.time_dependent:
          flags += ":fa:`clock-o` "
      if is_conditional:
          flags += ":fa:`question-circle` "
      if var.write_to_restart:
          flags += ":fa:`repeat` "

      print(f".. py:attribute:: VerosVariables.{key}")
      if key in seen:
          print("  :noindex:")

      print("")
      print(f"  :units: {var.units}")

      if var.dims:
          print(f"  :dimensions: {', '.join(var.dims)}")
      else:
          print(f"  :dimensions: scalar")

      print(f"  :type: :py:class:`{var.dtype or 'float'}`")

      if is_conditional:
          condition = inspect.getsource(var.active).strip()
          print(f"  :condition: ``{condition[7:-1]}``")

      print(f"  :attributes: {flags}")

      print("")
      print(f"  {var.long_description}")
      print("")
      seen.add(key)
