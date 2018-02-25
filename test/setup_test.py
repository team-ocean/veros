def test_setup_acc(backend):
    from veros.setup.acc import ACC
    sim = ACC(backend=backend)
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer * 20
    sim.run()


def test_setup_eady(backend):
    from veros.setup.eady import Eady
    sim = Eady(backend=backend)
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer * 20
    sim.run()


def test_setup_4deg(backend):
    from veros.setup.global_4deg import GlobalFourDegree
    sim = GlobalFourDegree(backend=backend)
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer * 20
    sim.run()


# too big to test
# def test_setup_1deg():
#     from veros.setup.global_1deg import GlobalOneDegree
#     sim = GlobalOneDegree()
#     sim.diskless_mode = True
#     sim.setup()
#     sim.runlen = sim.dt_tracer
#     sim.run()


def test_setup_north_atlantic(backend):
    from veros.setup.north_atlantic import NorthAtlantic
    sim = NorthAtlantic(backend=backend, override=dict(nx=100, ny=100, nz=50))
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer
    sim.run()


def test_setup_wave_propagation(backend):
    from veros.setup.wave_propagation import WavePropagation
    sim = WavePropagation(backend=backend, override=dict(nx=100, ny=100, nz=50))
    sim.diskless_mode = True
    sim.setup()
    sim.runlen = sim.dt_tracer
    sim.run()
