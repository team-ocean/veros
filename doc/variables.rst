Model variables
------------------

.. exec::
  import json
  from climate.pyom.variables import MAIN_VARIABLES
  #print("===== ===== =====")
  #print("Name unit description")
  #print("----- ----- -----")
  table_rows = []
  for key, var in MAIN_VARIABLES.items():
      table_rows.append([var.name, var.units, var.long_description])
      print("{} {} {}\n".format(key, var.units, var.long_description))
  #print("===== ===== =====")
