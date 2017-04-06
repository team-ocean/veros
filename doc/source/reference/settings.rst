Available settings
------------------

The default settings are handled differently from variables and diagnostics.

.. exec::
  from veros.settings import SETTINGS
  #headings = ["Setting", "Default value", "Description"]
  #print(".. csv-table:: Available settings")
  #print("   :header: {}".format(" ,".join(headings)))
  #print("   :widths: 10, 10, 20")
  #print("")
  for key, sett in SETTINGS.items():
      print(".. data:: {} = {}".format(key, sett.default))
      print("")
      print("   {}".format(sett.description))
      print("")
