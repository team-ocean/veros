import time
import logging

import click

from veros import VerosLegacy, veros_method, core, tools


class StreamfunctionBenchmark(VerosLegacy):
    def __init__(self, timesteps, *args, **kwargs):
        self.repetitions = timesteps
        super(StreamfunctionBenchmark, self).__init__(*args, **kwargs)

    @veros_method
    def set_parameter(self):
        self.identifier = "streamfunction_benchmark"
        self.diskless_mode = True

        m = self.main_module
        m.dt_mom = 480
        m.dt_tracer = 480

        m.coord_degree = 1
        m.enable_cyclic_x = 1

        m.congr_epsilon = 1e-12
        m.congr_max_iterations = 10000

        m.enable_congrad_verbose = 1

    @veros_method
    def set_grid(self):
        m = self.main_module
        m.dxt[:] = 80.0 / m.nx
        m.dyt[:] = 80.0 / m.ny
        m.dzt[:] = 5000. / m.nz
        m.x_origin = 0.0
        m.y_origin = -40.0

    @veros_method
    def set_coriolis(self):
        m = self.main_module
        m.coriolis_t[:, :] = 2 * m.omega * np.sin(m.yt[None, :] / 180. * np.pi)

    @veros_method
    def set_topography(self):
        m = self.main_module
        (X, Y) = np.meshgrid(m.xt, m.yt, indexing="ij")
        m.kbot[...] = (X > 1.0) | (Y < -20)

    @veros_method
    def set_initial_conditions(self):
        pass

    @veros_method
    def set_forcing(self):
        pass

    @veros_method
    def set_diagnostics(self):
        pass

    @veros_method
    def run(self):
        np.random.seed(123456789)
        for _ in range(self.repetitions):
            if self.legacy_mode:
                m = self.fortran.main_module
                rhs = np.zeros((m.ie_pe-m.is_pe+5, m.je_pe-m.js_pe+5), order="f", dtype=self.default_float_type)
                rhs[2:-2, 2:-2] = np.random.randn(m.ie_pe-m.is_pe+1, m.je_pe-m.js_pe+1)
                sol = np.zeros_like(rhs)
                start = time.time()
                self.fortran.congrad_streamfunction(
                    is_=m.is_pe-m.onx, ie_=m.ie_pe+m.onx, js_=m.js_pe-m.onx, je_=m.je_pe+m.onx,
                    forc=rhs, iterations=m.congr_itts, sol=sol, converged=False
                )
            else:
                rhs = np.zeros((self.nx+4, self.ny+4), dtype=self.default_float_type)
                rhs[2:-2, 2:-2] = np.random.randn(self.nx, self.ny)
                sol = np.zeros_like(rhs)
                start = time.time()
                core.external.solve_poisson.solve(self, rhs, sol)
            end = time.time()
            logging.info("Time step took {}s".format(end - start))

    @veros_method
    def after_timestep(self):
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
