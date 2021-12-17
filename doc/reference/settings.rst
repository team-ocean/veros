Model settings
------------------------

The following list of available settings is automatically created from the file :file:`settings.py` in the Veros main folder.
They are available as attributes of the :class:`Veros settings object <veros.state.VerosSettings>`, e.g.: ::

   >>> simulation = MyVerosSetup()
   >>> settings = simulation.state.settings
   >>> print(settings.eq_of_state_type)
   1

.. exec::
  from veros.settings import SETTINGS
  for key, sett in SETTINGS.items():
      print(".. _setting-{}:".format(key))
      print("")
      print(".. py:attribute:: VerosSettings.{} = {}".format(key, sett.default))
      print("")
      print("   {}".format(sett.description))
      print("")
