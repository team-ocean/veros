from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from veros.pyom_compat import pyom_from_state


def test_disabled_partial_cells_export(monkeypatch):
    state = SimpleNamespace(settings={"enable_partial_cells": False}, variables={}, diagnostics={})
    main = SimpleNamespace(tau=0, taup1=1, taum1=2)
    pyom = Mock(
        main_module=main,
        isoneutral_module=SimpleNamespace(),
        idemix_module=SimpleNamespace(),
        tke_module=SimpleNamespace(),
        eke_module=SimpleNamespace(),
    )
    monkeypatch.setattr("veros.pyom_compat._override_settings", lambda pyom: None)

    assert pyom_from_state(state, pyom, init_streamfunction=False) is pyom
    assert not hasattr(main, "enable_partial_cells")
    pyom.allocate_main_module.assert_called_once_with()


def test_enabled_partial_cells_rejected():
    state = SimpleNamespace(settings={"enable_partial_cells": True})
    with pytest.raises(ValueError, match="partial cells"):
        pyom_from_state(state, None)
