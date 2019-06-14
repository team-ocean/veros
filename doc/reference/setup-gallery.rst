Setup gallery
=============

This page gives an overview of the available model setups. To copy the setup file and additional input files (if applicable) to the current working directory, you can make use of the :command:`veros copy-setup` command.

Example::

   $ veros copy-setup acc


Idealized configurations
------------------------

+-------------------------------------------+-------------------------------------------+
| :doc:`/reference/setups/acc`              | :doc:`/reference/setups/acc_sector`       |
|                                           |                                           |
| |acc|                                     | |acc_sector|                              |
+-------------------------------------------+-------------------------------------------+

.. |acc| image:: /_images/gallery/acc.png
  :width: 100%
  :align: middle
  :target: setups/acc.html
  :alt: Steady-state stream function

.. |acc_sector| image:: /_images/gallery/acc_sector.png
  :width: 100%
  :scale: 75%
  :align: middle
  :target: setups/acc_sector.html
  :alt: Steady-state stream function

.. toctree::
   :hidden:

   setups/acc
   setups/acc_sector

Realistic configurations
------------------------

+--------------------------------------------+-------------------------------------------+
| :doc:`/reference/setups/flexible`          | :doc:`/reference/setups/wave-propagation` |
|                                            |                                           |
| |flexible|                                 | |wave-propagation|                        |
+--------------------------------------------+-------------------------------------------+
| :doc:`/reference/setups/4deg`              | :doc:`/reference/setups/1deg`             |
|                                            |                                           |
| |4deg|                                     | |1deg|                                    |
+--------------------------------------------+-------------------------------------------+
| :doc:`/reference/setups/north-atlantic`    |                                           |
|                                            |                                           |
| |northatlantic|                            |                                           |
+--------------------------------------------+-------------------------------------------+

.. |flexible| image:: /_images/gallery/flexible.png
   :width: 100%
   :align: middle
   :target: setups/flexible.html
   :alt: Surface velocity at 0.25x0.25 degree resolution

.. |wave-propagation| image:: /_images/gallery/wave-propagation.png
   :width: 100%
   :align: middle
   :target: setups/wave-propagation.html
   :alt: Stream function

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
   setups/wave-propagation
   setups/4deg
   setups/1deg
   setups/north-atlantic

Configurations with BioGeoChemistry
-----------------------------------

+--------------------------------------------+--------------------------------------------+
| :doc:`/reference/setups/bgc-4deg`          |                                            |
|                                            |                                            |
| |bgc-4deg|                                 |                                            |
+--------------------------------------------+--------------------------------------------+

.. |bgc-4deg| image:: /_images/gallery/bgc-4deg.png
   :width: 100%
   :align: middle
   :target: setups/bgc-4deg.html
   :alt: Surface zooplancton after 60 years

.. toctree::
   :hidden:

   setups/bgc-4deg
