def test_setup_acc(backend):
    from veros import runtime_settings as rs
    from veros.setup.acc import ACCSetup
    rs.backend = backend

    sim = ACCSetup()
    sim.state.diskless_mode = True
    sim.setup()
    sim.state.runlen = sim.state.dt_tracer * 20
    sim.run()


def test_setup_acc_sector(backend):
    from veros import runtime_settings as rs
    from veros.setup.acc_sector import ACCSectorSetup
    rs.backend = backend

    sim = ACCSectorSetup()
    sim.state.diskless_mode = True
    sim.setup()
    sim.state.runlen = sim.state.dt_tracer * 20
    sim.run()


def test_setup_4deg(backend):
    from veros import runtime_settings as rs
    from veros.setup.global_4deg import GlobalFourDegreeSetup
    rs.backend = backend

    sim = GlobalFourDegreeSetup()
    sim.state.diskless_mode = True
    sim.setup()
    sim.state.runlen = sim.state.dt_tracer * 20
    sim.run()


def test_setup_flexible(backend):
    from veros import runtime_settings as rs
    from veros.setup.global_flexible import GlobalFlexibleResolutionSetup
    rs.backend = backend

    sim = GlobalFlexibleResolutionSetup(override=dict(
        nx=100, ny=50, dt_tracer=3600, dt_mom=3600,
    ))
    sim.state.diskless_mode = True
    sim.setup()
    sim.state.runlen = sim.state.dt_tracer * 20
    sim.run()


def test_setup_1deg(backend):
    from veros import runtime_settings as rs
    from veros.setup.global_1deg import GlobalOneDegreeSetup
    rs.backend = backend

    sim = GlobalOneDegreeSetup()
    sim.state.diskless_mode = True
    # too big to test
    # sim.setup()
    # sim.state.runlen = sim.state.dt_tracer
    # sim.run()


def test_setup_north_atlantic(backend):
    from veros import runtime_settings as rs
    from veros.setup.north_atlantic import NorthAtlanticSetup
    rs.backend = backend

    sim = NorthAtlanticSetup(override=dict(nx=100, ny=100, nz=50))
    sim.state.diskless_mode = True
    sim.setup()
    sim.state.runlen = sim.state.dt_tracer
    sim.run()


def test_setup_wave_propagation(backend):
    from veros import runtime_settings as rs
    from veros.setup.wave_propagation import WavePropagationSetup
    rs.backend = backend

    sim = WavePropagationSetup(override=dict(nx=100, ny=100, nz=50))
    sim.state.diskless_mode = True
    sim.setup()
    sim.state.runlen = sim.state.dt_tracer
    sim.run()
