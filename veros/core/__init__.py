import os
import importlib

from loguru import logger


def build_all():
    """Trigger import of all core modules"""
    basedir = os.path.dirname(__file__)

    for root, dirs, files in os.walk(basedir):
        py_path = '.'.join(os.path.split(os.path.relpath(root, basedir))).strip('.')

        for f in files:
            modname, ext = os.path.splitext(f)
            if modname.startswith('__') or ext != '.py':
                continue

            if py_path:
                module_path = f'veros.core.{py_path}.{modname}'
            else:
                module_path = f'veros.core.{modname}'

            logger.trace('importing {}', module_path)
            importlib.import_module(module_path)
