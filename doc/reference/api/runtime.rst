Runtime settings & state
========================


Runtime settings
----------------

.. autoclass:: veros.runtime.RuntimeSettings
   :members:
   :undoc-members:

Persistent JAX compilation cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``jax_compilation_cache_dir`` before importing Veros core modules to reuse
compiled executables across processes::

    from veros import runtime_settings

    runtime_settings.backend = "jax"
    runtime_settings.jax_compilation_cache_dir = "/path/to/veros-cache"

Set ``VEROS_JAX_COMPILATION_CACHE_DIR`` when launching a setup from the command
line::

    VEROS_JAX_COMPILATION_CACHE_DIR=/path/to/veros-cache veros run acc.py -b jax

The default is an empty string, which leaves JAX's own cache configuration
unchanged. Only the JAX backend uses this setting. JAX retains its usual cache
keys and storage thresholds; a warm cache reduces compilation time, rather than
execution time for an already compiled kernel. Choose a directory writable only
by trusted users, since JAX loads executable code from the cache.

See the `JAX persistent cache documentation
<https://docs.jax.dev/en/latest/persistent_compilation_cache.html>`_ for size
limits, minimum compilation times, and distributed cache configuration.

Runtime state
-------------

.. autoclass:: veros.runtime.RuntimeState
   :members:
   :undoc-members:
