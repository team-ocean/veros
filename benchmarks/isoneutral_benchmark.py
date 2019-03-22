import time

import click

from veros import (
    VerosLegacy, veros_method, core, tools, runtime_settings as rs,
    runtime_state as rst
)
from veros.distributed import barrier


class IsoneutralBenchmark(VerosLegacy):
    def __init__(self, timesteps, *args, **kwargs):
        self.repetitions = timesteps
        super(IsoneutralBenchmark, self).__init__(*args, **kwargs)

    @veros_method
    def set_parameter(self, vs):
        np.random.seed(123456789)
        m = self.main_module
        m.dt_tracer = m.dt_mom = 86400.
        m.enable_cyclic_x = True
        m.eq_of_state_type = 3
        m.congr_epsilon = 1
        m.congr_max_iterations = 1

        im = self.isoneutral_module
        im.enable_neutral_diffusion = True
        im.iso_slopec = 0
        im.K_iso_0 = np.random.rand()
        im.K_iso_steep = np.random.rand()
        im.iso_dslope = np.random.rand()

    @veros_method
    def set_grid(self, vs):
        if not self.legacy_mode:
            vs.is_pe = vs.js_pe = 1
            vs.ie_pe, vs.je_pe = vs.nx // rs.num_proc[0], vs.ny // rs.num_proc[1]

        m = self.main_module
        self.set_attribute(vs, "x_origin", np.random.rand())
        self.set_attribute(vs, "y_origin", np.random.rand())
        self.set_attribute(vs, "dxt", 1 + 100 * np.random.rand(m.ie_pe-m.is_pe+5).astype(vs.default_float_type))
        self.set_attribute(vs, "dyt", 1 + 100 * np.random.rand(m.je_pe-m.js_pe+5).astype(vs.default_float_type))
        self.set_attribute(vs, "dzt", 1 + 100 * np.random.rand(m.nz).astype(vs.default_float_type))

    @veros_method
    def set_topography(self, vs):
        m = self.main_module
        kbot = np.zeros((m.ie_pe-m.is_pe+5,m.je_pe-m.js_pe+5))
        kbot[2:-2, 2:-2] = np.random.randint(1, m.nz, size=(m.ie_pe-m.is_pe+1,m.je_pe-m.js_pe+1))
        self.set_attribute(vs, "kbot", kbot)

    @veros_method
    def set_coriolis(self, vs):
        m = self.main_module
        self.set_attribute(vs, "coriolis_t", 2 * np.random.rand(1, m.je_pe-m.js_pe+5).astype(vs.default_float_type) - 1.)

    @veros_method
    def set_initial_conditions(self, vs):
        m = self.main_module
        self.set_attribute(vs, "salt", 35 + np.random.randn(m.ie_pe - m.is_pe + 5, m.je_pe - m.js_pe + 5, m.nz, 3).astype(vs.default_float_type))
        self.set_attribute(vs, "temp", 20 + 5 * np.random.rand(m.ie_pe - m.is_pe + 5, m.je_pe - m.js_pe + 5, m.nz, 3).astype(vs.default_float_type))

    @veros_method
    def set_forcing(self, vs):
        m = self.main_module
        for a in ("flux_east","flux_north","flux_top","u_wgrid","v_wgrid","w_wgrid","K_iso","K_gm","du_mix","P_diss_iso","P_diss_skew"):
            self.set_attribute(vs, a,np.random.randn(m.ie_pe-m.is_pe+5,m.je_pe-m.js_pe+5,m.nz).astype(vs.default_float_type))

    def set_diagnostics(self, vs):
        pass

    @veros_method
    def set_attribute(self, vs, attribute, value):
        if self.legacy_mode:
            legacy_modules = ("main_module", "isoneutral_module", "tke_module",
                              "eke_module", "idemix_module")
            for module in legacy_modules:
                module_handle = getattr(self, module)
                if hasattr(module_handle, attribute):
                    try:
                        v = np.asfortranarray(value).copy2numpy()
                    except AttributeError:
                        v = np.asfortranarray(value)
                    getattr(module_handle, attribute)[...] = v
                    assert np.all(value == getattr(module_handle, attribute)), attribute
                    return
            raise AttributeError("Legacy pyOM has no attribute {}".format(attribute))
        else:
            if isinstance(value, np.ndarray):
                getattr(vs, attribute)[...] = value
            else:
                setattr(vs, attribute, value)

    def run(self):
        vs = self.state
        for t in range(self.repetitions):
            start = time.time()
            if self.legacy_mode:
                self.fortran.isoneutral_diffusion_pre()
            else:
                core.isoneutral.isoneutral_diffusion_pre(vs)
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
    sim = IsoneutralBenchmark(*args, **kwargs)
    sim.setup()
    sim.run()


if __name__ == "__main__":
    main()
