def test_setup_acc():
    from veros.setup.acc import ACC
    sim = ACC()
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer * 20
    sim.run()


def test_setup_eady():
    from veros.setup.eady import Eady
    sim = Eady()
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer * 20
    sim.run()


def test_setup_4deg():
    from veros.setup.global_4deg import GlobalFourDegree
    sim = GlobalFourDegree()
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer * 20
    sim.run()


# def test_setup_1deg():
#     from veros.setup.global_1deg import GlobalOneDegree
#     sim = GlobalOneDegree()
#     sim.diskless_mode = True
#     sim.setup()
#     sim.runlen = sim.dt_tracer
#     sim.run()


def test_setup_north_atlantic():
    from veros.setup.north_atlantic import NorthAtlantic
    sim = NorthAtlantic(override=dict(nx=100, ny=100, nz=50))
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer
    sim.run()


def test_setup_wave_propagation():
    from veros.setup.wave_propagation import WavePropagation
    sim = WavePropagation(override=dict(nx=100, ny=100, nz=50))
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer
    sim.run()
