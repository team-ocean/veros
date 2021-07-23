How to write a Veros plug-in
============================

Writing a plug-in for Veros is relatively simple.
A plug-in can be any Python package that accepts a :class:`VerosState <veros.state.VerosState>` object.
The plug-in is then free to modify the model state in any way it pleases.

The only requirement for the plug-in to be usable in Veros setups is that it declares a special object in its top :file:`__init__.py` file:

::

   __VEROS_INTERFACE__ = dict(
       name='my-plugin',
       setup_entrypoint=my_setup_function,
       run_entrypoint=my_main_function
   )

The functions passed in :obj:`setup_entrypoint` and :obj:`run_entrypoint` are then called by Veros during model set-up and after each time step, respectively, with the model state as the sole argument.

An example for these functions could be:

::

   from veros import veros_routine


   @veros_routine
   def my_setup_function(state):
       pass


   @veros_routine
   def my_main_function(state):
       from veros.core.operators import numpy as npx
       vs = state.variables

       # apply simple ice mask
       mask = np.logical_and(vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8,
                             vs.forc_temp_surface < 0.)
       vs.forc_temp_surface = npx.where(mask, 0.0, vs.forc_temp_surface)
       vs.forc_salt_surface = npx.where(mask, 0.0, vs.forc_salt_surface)


In this case, the setup function does nothing, while the main function sets temperature and salinity forcing to 0 where the surface temperature is smaller than -1.8 degrees (a very crude sea ice model).

Custom settings, variables, and diagnostics
-------------------------------------------

In real-world applications, you probably want to use custom settings, variables, and/or diagnostics in your plug-in.
You can specify those as additional arguments to :obj:`__VEROS_INTERFACE__`:

::

   __VEROS_INTERFACE__ = dict(
       name='my-plugin',
       setup_entrypoint=my_setup_function,
       run_entrypoint=my_main_function,
       settings=my_settings,
       variables=my_variables,
       diagnostics=[MyDiagnostic]
   )

In this case, :obj:`my_settings` is a :class:`dict` mapping the name of the setting to a :class:`Setting <veros.settings.Setting>` object:

::

   from veros.settings import Setting

   my_settings = {
       'enable_my_plugin'; Setting(False, bool, 'Enable my plugin'),
       'temperature_cutoff': Setting(-1.8, float, 'Cut-off surface temperature'),
   }


Similarly, for variables:

::

   from veros.variables import Variable, T_GRID

   my_variables = {
       'my_variable': Variable('Description', T_GRID, 'unit', 'Long description'),
   }

The so-defined settings and variables are then available as attributes of the Veros state object, as usual:

::

   @veros_routine
   def my_function(state):
       from veros.core.operators import update, at

       if state.settings.enable_my_plugin:
           state.variables.my_variable = update(state.variables.my_variable, at[...], 0.)

.. seealso::

   For more inspiration on how to specify settings and variables, have a look at the built-in :file:`settings.py` and :file:`variables.py` files.

Diagnostics are defined similarly, but they have to be a subclass of :class:`VerosDiagnostic <veros.diagnostics.base.VerosDiagnostic>`.


Shipping custom model setups
----------------------------

You can use a special entrypoint in the :file:`setup.py` file of your plug-in to inform the Veros command-line interface of your custom setups:

::

   from setuptools import setup

   setup(
      name='my-plugin',
      packages='my_plugin',
      entry_points={
        'veros.setup_dirs': [
            'my_plugin = my_plugin.setups'
        ]
      }
   )

This assumes, that your custom setups are located in the folder :file:`my_plugin/setups`.
Then, `veros copy-setup` will automatically find your custom setups if the plug-in is installed:

::

   $ veros copy-setup --help
   Usage: veros copy-setup [OPTIONS] SETUP

   Copy a standard setup to another directory.

   Available setups:

      acc, acc_basic, acc_sector, eady, fjord, global_1deg, global_4deg,
      global_flexible, my_setup, north_atlantic, wave_propagation

   Example:

      $ veros copy-setup global_4deg --to ~/veros-setups/4deg-lowfric

   Further directories containing setup templates can be added to this
   command via the VEROS_SETUP_DIR environment variable.

   Options:
   --to PATH  Target directory, must not exist (default: copy to current
              working directory)
   --help     Show this message and exit.

In this case, the custom setup is located in the folder :file:`my_plugin/setups/my_setup`, and thus shows up as :obj:`my_setup`.
