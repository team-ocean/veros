import pytest

from veros.plugins import load_plugin
from veros.routines import veros_routine
from veros.state import get_default_state
from veros.variables import Variable
from veros.settings import Setting


@pytest.fixture
def fake_plugin():
    class FakePlugin:
        pass

    def run_setup(state):
        plugin._setup_ran = True

    def run_main(state):
        plugin._main_ran = True

    plugin = FakePlugin()
    plugin.__name__ = "foobar"
    plugin._setup_ran = False
    plugin._main_ran = False
    plugin.__VEROS_INTERFACE__ = {
        "name": "foo",
        "setup_entrypoint": run_setup,
        "run_entrypoint": run_main,
        "settings": dict(mydimsetting=Setting(15, int, "bar")),
        "variables": dict(myvar=Variable("myvar", ("xt", "yt", "mydim"))),
        "dimensions": dict(mydim="mydimsetting"),
        "diagnostics": [],
    }
    yield plugin


def test_load_plugin(fake_plugin):
    plugin_interface = load_plugin(fake_plugin)
    assert plugin_interface.name == "foo"


def test_state_plugin(fake_plugin):
    plugin_interface = load_plugin(fake_plugin)
    state = get_default_state(plugin_interfaces=plugin_interface)
    assert "mydimsetting" in state.settings
    assert "mydim" in state.dimensions
    assert state.dimensions["mydim"] == state.settings.mydimsetting
    state.initialize_variables()
    assert "myvar" in state.variables
    assert state.variables.myvar.shape == (4, 4, state.settings.mydimsetting)


def test_run_plugin(fake_plugin):
    from veros.setups.acc_basic import ACCBasicSetup

    class FakeSetup(ACCBasicSetup):
        __veros_plugins__ = (fake_plugin,)

        @veros_routine
        def set_diagnostics(self, state):
            pass

    setup = FakeSetup(override=dict(dt_tracer=100, runlen=100))

    assert not fake_plugin._setup_ran
    setup.setup()
    assert fake_plugin._setup_ran

    assert not fake_plugin._main_ran
    setup.run()
    assert fake_plugin._main_ran
