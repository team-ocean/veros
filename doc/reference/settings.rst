Available settings
------------------

The following list of available settings is automatically created from the file :file:`settings.py` in the Veros main folder. They are available as attributes of all instances of the :class:`Veros main class <veros.Veros>`, e.g.: ::

   >>> simulation = MyVerosClass()
   >>> print(simulation.eq_of_state_type)
   1

.. exec::
  from veros.settings import SETTINGS
  for key, sett in SETTINGS.items():
      print(".. _setting-{}:".format(key))
      print("")
      print(".. data:: {} = {}".format(key, sett.default))
      print("")
      print("   {}".format(sett.description))
      print("")
