Available model settings
------------------------

The following list of available settings is automatically created from the file :file:`settings.py` in the Veros main folder.
They are available as attributes of all instances of the :class:`Veros state class <veros.VerosState>`, e.g.: ::

   >>> simulation = MyVerosSetup()
   >>> vs = simulation.state
   >>> print(vs.eq_of_state_type)
   1

.. exec::
  from veros.settings import SETTINGS
  for key, sett in SETTINGS.items():
      print(".. _setting-{}:".format(key))
      print("")
      print(".. py:attribute:: VerosState.{} = {}".format(key, sett.default))
      print("")
      print("   {}".format(sett.description))
      print("")
