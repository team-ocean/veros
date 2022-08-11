from collections import namedtuple

from veros.variables import Variable
from veros.settings import Setting

VerosPlugin = namedtuple(
    "VerosPlugin",
    [
        "name",
        "module",
        "setup_entrypoint",
        "run_entrypoint",
        "settings",
        "variables",
        "dimensions",
        "diagnostics",
    ],
)


def load_plugin(module):
    from veros.diagnostics.base import VerosDiagnostic

    modname = module.__name__

    if not hasattr(module, "__VEROS_INTERFACE__"):
        raise RuntimeError(f"module {modname} is not a valid Veros plugin")

    interface = module.__VEROS_INTERFACE__

    setup_entrypoint = interface.get("setup_entrypoint")

    if not callable(setup_entrypoint):
        raise RuntimeError(f"module {modname} is missing a valid setup entrypoint")

    run_entrypoint = interface.get("run_entrypoint")

    if not callable(run_entrypoint):
        raise RuntimeError(f"module {modname} is missing a valid run entrypoint")

    name = interface.get("name", modname)

    settings = interface.get("settings", {})
    for setting, val in settings.items():
        if not isinstance(val, Setting):
            raise TypeError(f"got unexpected type {type(val)} for setting {setting}")

    variables = interface.get("variables", {})
    for variable, val in variables.items():
        if not isinstance(val, Variable):
            raise TypeError(f"got unexpected type {type(val)} for variable {variable}")

    dimensions = interface.get("dimensions", {})
    for dim, val in dimensions.items():
        if not isinstance(val, (str, int)):
            raise TypeError(f"got unexpected type {type(val)} for dimension {dim}")

    diagnostics = interface.get("diagnostics", [])
    for diagnostic in diagnostics:
        if not issubclass(diagnostic, VerosDiagnostic):
            raise TypeError(f"got unexpected type {type(diagnostic)} for diagnostic {diagnostic}")

    return VerosPlugin(
        name=name,
        module=module,
        setup_entrypoint=setup_entrypoint,
        run_entrypoint=run_entrypoint,
        settings=settings,
        variables=variables,
        dimensions=dimensions,
        diagnostics=diagnostics,
    )
