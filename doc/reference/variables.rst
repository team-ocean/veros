.. _variables:

Model variables
===============

The variable meta-data (i.e., all instances of :class:`~veros.variables.Variable`)
are available in a dictionary as the attribute :attr:`VerosState.var_meta <veros.state.VerosState.var_meta>`. The actual
data arrays are attributes of :attr:`VerosState.variables <veros.state.VerosState.variables>`:

::

   state.variables.psi  # data array for variable psi
   state.var_meta["psi"]  # metadata for variable psi

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
  from veros.variables import VARIABLES

  def format_field(val):
      import inspect

      if isinstance(val, (tuple, list)):
          return "(" + ", ".join(map(str, val)) + ")"

      if not callable(val):
          return val

      src = inspect.getsource(val)
      src = src.strip().rstrip(",")
      return f"``{src}``"

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
      print(f"  :units: {format_field(var.units)}")

      if var.dims is not None:
          print(f"  :dimensions: {format_field(var.dims)}")
      else:
          print(f"  :dimensions: scalar")

      print(f"  :type: :py:class:`{format_field(var.dtype) or 'float'}`")

      if is_conditional:
          condition = format_field(var.active).replace("active=", "")
          print(f"  :condition: {condition}")

      print(f"  :attributes: {flags}")

      print("")
      print(f"  {format_field(var.long_description)}")
      print("")
      seen.add(key)
