Making changes in Veros
=======================

Code conventions
----------------

When contributing to Veros, please adhere to the following general guidelines:

- Your first guide should be the surrounding Veros code. Look around, and be consistent with your modifications.
- Unless you have a very good reason not to do so, please stick to `the PEP8 style guide <https://www.python.org/dev/peps/pep-0008/>`_ throughout your code. One exception we make in Veros is in regard to the maximum line length - since numerical operations can take up quite a lot of horizontal space, you may use longer lines if it increases readability.
- Please follow the PEP8 naming conventions, and use meaningful, telling names for your variables, functions, and classes. The variable name :data:`stretching_factor` is infinitely more meaningful than :data:`k`. This is especially important for settings and generic helper functions.
- "Private" helper functions that are not meant to be called from outside the current source file should be prefixed with an underscore (``_``).
- Use double quotes (``"``) for all strings longer than a single character.
- Document your functions using `Google-style docstrings <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_. This is especially important if you are implementing a user-facing API (such as a diagnostic, a setup, or tools that are meant to be called from setups).

Distributed memory support
--------------------------

By default, all core routines should support distributed execution via MPI.
In this case, every processor only operates on a chunk of the total data.
By using :py:func:`veros.variables.allocate`, you can make sure that allocated data always has the right shape.

Since none of the processes have access to the global data, you need to take special care during reductions (e.g. ``sum``) and accumulations (e.g. ``cumsum``) along horizontal dimensions.
Use functions from :mod:`veros.distributed` (e.g. :func:`veros.distributed.global_max`) where appropriate.

The dist_safe keyword
+++++++++++++++++++++

If you are not comfortable writing code that is safe for distributed execution, you can use the ``dist_safe`` keyword to :func:`veros.decorators.veros_method`:::

   @veros_method(dist_safe=False, local_variables=["temp"])
   def my_function(vs):
       # this function is now guaranteed to be executed on the main process

       # since temp is declared as a local variable, we have access to all of the data
       vs.temp[2:-2, 2:-2] = np.max(vs.temp)

       # this would throw an error, since salt is not in local_variables
       # vs.salt[...] = 0

       # after execution, the updated contents of vs.temp are scattered to all processes,
       # and distributed execution continues

When encountering a ``veros_method`` that is marked as not safe for distributed execution (``dist_safe=False``), Veros gathers all relevant data from the worker processes,
copies it to the main process, and executes the function there.
This ensures that you can write your code exactly as in the non-distributed case (but it comes with a performance penalty, of course).

Running tests and benchmarks
----------------------------

If you want to make sure that your changes did not break anything, you can run our test suite that compares the results of each subroutine to pyOM2.
To do that, you will need to compile the Python interface of pyOM2 on your machine, and then point the testing suite to the library location, e.g. through::

   $ pytest -v . --pyom2-lib /path/to/pyOM2/py_src/pyOM_code.so

from the main folder of the Veros repository.

If you deliberately introduced breaking changes, you can disable them during testing by prefixing them with::

   if not vs.pyom_compatibility_mode:
       # your changes

Automated benchmarks are provided in a similar fashion. The benchmarks run some dummy problems with varying problem sizes and all available computational backends: ``numpy``, ``bohrium-openmp``, ``bohrium-opencl``, ``bohrium-cuda``, ``fortran`` (pyOM2), and ``fortran-mpi`` (parallel pyOM2). For options and further information run::

   $ python run_benchmarks.py --help

from the :file:`test` folder. Timings are written in YAML format.

Performance tweaks
------------------

If your changes to Veros turn out to have a negative effect on the runtime of the model, there several ways to investigate and solve performance problems:

- Run your model with the ``-v debug`` option to get additional debugging output (such as timings for each time step, and a timing summary after the run has finished).
- Run your model with the ``-p`` option to profile Veros with pyinstrument. You may have to run :command:`pip install pyinstrument` before being able to do so. After completion of the run, a file :file:`profile.html` will be written that can be opened with a web browser and contains timings for the entire call stack.
- You should try and avoid explicit loops over arrays at all cost (even more so when using Bohrium). You should always try to work on the whole array at once.
- When using Bohrium, it is sometimes beneficial to copy an array to NumPy before passing it to an external module or performing an operation that cannot be vectorized efficiently. Just don't forget to copy it back to Bohrium after you are finished, e.g. like so: ::

      from veros import runtime_settings as rs

      if rs.backend == "bohrium":
          u_np = vs.u.copy2numpy()
      else:
          u_np = vs.u

      vs.u[...] = np.asarray(external_function(u_np))

- If you are still having trouble, don't hesitate to ask for help (e.g. `on GitHub <https://github.com/team-ocean/veros/issues>`_).
