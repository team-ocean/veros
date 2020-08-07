import time

import click

from veros import VerosLegacy, veros_method, tools, runtime_settings as rs, runtime_state as rst
from veros.distributed import barrier


class StreamfunctionBenchmark(VerosLegacy):
    def __init__(self, timesteps, *args, **kwargs):
        self.repetitions = timesteps
        super(StreamfunctionBenchmark, self).__init__(*args, **kwargs)

    @veros_method
    def set_parameter(self, vs):
        vs.identifier = "streamfunction_benchmark"
        vs.diskless_mode = True

        m = self.main_module
        m.dt_mom = 480
        m.dt_tracer = 480

        m.coord_degree = 1
        m.enable_cyclic_x = 1

        m.congr_epsilon = 1e-12
        m.congr_max_iterations = 10000

        m.enable_congrad_verbose = 1

    @veros_method
    def set_grid(self, vs):
        m = self.main_module
        m.dxt[:] = 80.0 / m.nx
        m.dyt[2:-2] = tools.get_vinokur_grid_steps(
            m.ny, 80., 0.66 * 80. / m.ny, two_sided_grid=True
        )[m.js_pe-1:m.je_pe]
        m.dzt[:] = 5000. / m.nz
        m.x_origin = 0.0
        m.y_origin = -40.0

    @veros_method
    def set_coriolis(self, vs):
        m = self.main_module
        m.coriolis_t[:, :] = 2 * m.omega * np.sin(m.yt[None, :] / 180. * np.pi)

    @veros_method
    def set_topography(self, vs):
        m = self.main_module
        (X, Y) = np.meshgrid(m.xt, m.yt, indexing="ij")
        bottom_slope = np.linspace(m.nz - 1, 1, m.nx)[m.is_pe-1:m.ie_pe]
        m.kbot[2:-2] = bottom_slope.reshape(-1, 1).astype('int')
        m.kbot[...] *= (X > 1.0) | (Y < -20)

        island_mask = np.ones((m.nx + 4, m.ny + 4), dtype='int')
        island_width = m.nx // 5
        island_start = 2 * m.nx // 5
        island_end = island_start + island_width
        island_mask[island_start:island_end, island_start:island_end] = 0
        m.kbot[2:-2, 2:-2] *= island_mask[m.is_pe-1:m.ie_pe, m.js_pe-1:m.je_pe]

    @veros_method
    def set_initial_conditions(self, vs):
        pass

    @veros_method
    def set_forcing(self, vs):
        pass

    @veros_method
    def set_diagnostics(self, vs):
        pass

    def run(self):
        from veros import runtime_state
        vs = self.state
        np = runtime_state.backend_module

        np.random.seed(123456789)
        for _ in range(self.repetitions):
            if self.legacy_mode:
                m = self.fortran.main_module
                rhs = np.zeros((m.ie_pe-m.is_pe+5, m.je_pe-m.js_pe+5), order="f", dtype=vs.default_float_type)
                rhs[2:-2, 2:-2] = np.random.randn(m.ie_pe-m.is_pe+1, m.je_pe-m.js_pe+1)
                sol = np.zeros_like(rhs)
                start = time.time()
                self.fortran.congrad_streamfunction(
                    is_=m.is_pe-m.onx, ie_=m.ie_pe+m.onx, js_=m.js_pe-m.onx, je_=m.je_pe+m.onx,
                    forc=rhs, iterations=m.congr_itts, sol=sol, converged=False
                )
            else:
                rhs = np.zeros((vs.nx // rs.num_proc[0] + 4, vs.ny // rs.num_proc[1] + 4), dtype=vs.default_float_type)
                rhs[2:-2, 2:-2] = np.random.randn(*rhs[2:-2, 2:-2].shape)
                sol = np.zeros_like(rhs)
                start = time.time()
                vs.linear_solver.solve(vs, rhs, sol)
            barrier()
            if rst.proc_rank == 0:
                print("Time step took {:.2e}s".format(time.time() - start))

    @veros_method
    def after_timestep(self, vs):
        pass


@click.option('-f', '--fortran', type=click.Path(exists=True), default=None)
@click.option('--timesteps', type=int, default=100)
@tools.cli
def main(*args, **kwargs):
    sim = StreamfunctionBenchmark(*args, **kwargs)
    sim.setup()
    sim.run()


if __name__ == "__main__":
    main()
