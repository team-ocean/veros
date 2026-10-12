import pytest

from veros.runtime import RuntimeSettings


def test_automatic_tdma_default(monkeypatch):
    monkeypatch.delenv("VEROS_USE_SPECIAL_TDMA", raising=False)
    assert RuntimeSettings().use_special_tdma is None


@pytest.mark.parametrize("value, expected", ((None, None), (True, True), (False, False)))
def test_tdma_override(value, expected, monkeypatch):
    monkeypatch.setenv("VEROS_USE_SPECIAL_TDMA", "1")
    assert RuntimeSettings(use_special_tdma=value).use_special_tdma is expected


@pytest.mark.parametrize("value, expected", (("1", True), ("0", False)))
def test_tdma_environment(value, expected, monkeypatch):
    monkeypatch.setenv("VEROS_USE_SPECIAL_TDMA", value)
    assert RuntimeSettings().use_special_tdma is expected
