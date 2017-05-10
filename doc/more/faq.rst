Frequently asked questions
==========================

.. _when-to-use-bohrium:

Which backend should I choose to run my model (NumPy / Bohrium)?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Because in its current state Bohrium carries some computational overhead, this
mostly depends on your problem size and the architecture you want to use. As a rule
of thumb, switching from NumPy to Bohrium is beneficial if your set-up contains
at least 1,000,000 elements (total number of elements in a 3-dimensional array,
i.e., :math:`n_x n_y n_z`).
