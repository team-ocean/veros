Setup gallery
=============

This page gives an overview of the available model setups. To copy the setup file and additional input files (if applicable) to the current working directory, you can make use of the :command:`veros copy-setup` command.

Example::

   $ veros copy-setup acc


.. note::

   More setups are available through the `extra setups plugin <https://veros-extra-setups.readthedocs.io>`_.


Idealized configurations
------------------------

+-------------------------------------------+-------------------------------------------+
| :doc:`/reference/setups/acc`              | :doc:`/reference/setups/acc_basic`        |
|                                           |                                           |
| |acc|                                     | |acc_basic|                               |
+-------------------------------------------+-------------------------------------------+

.. |acc| image:: /_images/gallery/acc.png
  :width: 100%
  :align: middle
  :target: setups/acc.html
  :alt: Steady-state stream function

.. |acc_basic| image:: /_images/gallery/acc_basic.png
  :width: 100%
  :align: middle
  :target: setups/acc_basic.html
  :alt: Steady-state stream function


.. toctree::
   :hidden:

   setups/acc
   setups/acc_basic

Realistic configurations
------------------------

+--------------------------------------------+-------------------------------------------+
| :doc:`/reference/setups/flexible`          | :doc:`/reference/setups/4deg`             |
|                                            |                                           |
| |flexible|                                 | |4deg|                                    |
+--------------------------------------------+-------------------------------------------+
| :doc:`/reference/setups/1deg`              | :doc:`/reference/setups/north-atlantic`   |
|                                            |                                           |
| |1deg|                                     | |northatlantic|                           |
+--------------------------------------------+-------------------------------------------+

.. |flexible| image:: /_images/gallery/flexible.png
   :width: 100%
   :align: middle
   :target: setups/flexible.html
   :alt: Surface velocity at 0.25x0.25 degree resolution

.. |northatlantic| image:: /_images/gallery/north-atlantic.png
   :width: 100%
   :align: middle
   :target: setups/north-atlantic.html
   :alt: Resulting average surface speed

.. |4deg| image:: /_images/gallery/4deg.png
   :width: 100%
   :align: middle
   :target: setups/4deg.html
   :alt: Stream function after 50 years

.. |1deg| image:: /_images/gallery/1deg.png
   :width: 100%
   :align: middle
   :target: setups/1deg.html
   :alt: Stream function



.. toctree::
   :hidden:

   setups/flexible
   setups/4deg
   setups/1deg
   setups/north-atlantic
