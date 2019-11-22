How to use plug-ins
===================

A plug-in is an optional extension for Veros that you can use to enable additional physics, outputs, or diagnostics.

.. note::

   Plug-ins do not necessarily meet the same quality standards as the Veros core, and not all plug-ins are directly affiliated with Veros or the Veros developers. You should carefully check whether a given plug-in fits your needs.

As a first step, install the plug-in you want to use, e.g. :obj:`veros-bgc` via

::

   $ pip install veros-bgc


You can then immediately use any custom setups included in the plug-in:

::

   $ veros copy-setup bgc_global_4deg


To use a plug-in in a Veros setup, all you need to do is to import it and add it to the :obj:`__veros_plugins__` attribute of your setup class:

::

   import veros_bgc

   class MySetup(VerosSetup):
      __veros_plugins__ = (veros_bgc,)

      # - rest of the setup definition -

This step is probably not necessary if you use a setup that was shipped with the plug-in, but in doubt you should double-check that the plug-in is activated properly.

.. seealso::

   For more information, refer to the documentation of the plug-in in question. You can find some suggestions in the contents of this section.
