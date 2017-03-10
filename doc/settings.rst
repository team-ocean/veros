Available settings
------------------

.. exec::
  import json
  from climate.pyom.settings import SETTINGS
  json_obj = json.dumps(SETTINGS, sort_keys=True, indent=4)
  print '.. code-block:: JavaScript\n\n    %s\n\n' % json_obj
