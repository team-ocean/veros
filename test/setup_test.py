import pytest


@pytest.fixture(autouse=True)
def set_options():
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_acc(float_type):
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "float_type", float_type)

    from veros.setups.acc import ACCSetup

    sim = ACCSetup()
    sim.setup()

    with sim.state.settings.unlock():
        sim.state.settings.runlen = sim.state.settings.dt_tracer * 20

    sim.run()


def test_setup_acc_basic():
    from veros.setups.acc_basic import ACCBasicSetup

    sim = ACCBasicSetup()
    sim.setup()

    with sim.state.settings.unlock():
        sim.state.settings.runlen = sim.state.settings.dt_tracer * 20

    sim.run()


def test_setup_4deg():
    from veros.setups.global_4deg import GlobalFourDegreeSetup

    sim = GlobalFourDegreeSetup()
    sim.setup()

    with sim.state.settings.unlock():
        sim.state.settings.runlen = sim.state.settings.dt_tracer * 20

    sim.run()


def test_setup_flexible():
    from veros.setups.global_flexible import GlobalFlexibleResolutionSetup

    sim = GlobalFlexibleResolutionSetup(
        override=dict(
            nx=100,
            ny=50,
            dt_tracer=3600,
            dt_mom=3600,
        )
    )
    sim.setup()

    with sim.state.settings.unlock():
        sim.state.settings.runlen = sim.state.settings.dt_tracer * 20

    sim.run()


def test_setup_1deg():
    from veros.setups.global_1deg import GlobalOneDegreeSetup

    # too big to test
    GlobalOneDegreeSetup()


def test_setup_north_atlantic():
    from veros.setups.north_atlantic import NorthAtlanticSetup

    sim = NorthAtlanticSetup(override=dict(nx=100, ny=100, nz=50))
    sim.setup()

    with sim.state.settings.unlock():
        sim.state.settings.runlen = sim.state.settings.dt_tracer

    sim.run()
