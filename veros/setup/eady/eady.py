import veros
import veros.tools


class Eady(veros.Veros):
    """Demonstrates the classical Eady (1949) solution.

    A narrow channel on an f-plane with prescribed stratification and vertically sheared background zonal flow.
    The temperature variable of the model is in this case a temperature perturbation,
    the effect of the background stratification on the perturbation temperature is implemented in
    the configuration routine set_forcing.

    This setup demonstrates:
     - setting up a highly idealized configuration
     - adding additional variables to the model
     - adding additional variables to snapshot output
     - calling core routines from the setup

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/Eady%201>`_.
    """
    @veros.veros_method
    def set_parameter(self):
        self.identifier = "eady"

        self.nx, self.ny, self.nz = 32, 32, 20
        self.dt_tracer = 1200.
        self.dt_mom = 1200.
        self.runlen = 100 * 86400.

        self.congr_epsilon = 1e-12
        self.congr_max_iterations = 5000

        self.enable_cyclic_x = True

        self.enable_superbee_advection = True
        self.enable_explicit_vert_friction = True
        self.enable_hor_friction = True
        self.A_h = (20e3)**3 * 2e-11
        self.kappaM_0 = 1.e-4
        self.kappaH_0 = 1.e-4

        self.enable_conserve_energy = False
        self.coord_degree = False
        self.eq_of_state_type = 1
        self.enable_tempsalt_sources = True

    @veros.veros_method
    def set_grid(self):
        self.x_origin = 0.
        self.y_origin = 0.
        self.dxt[...] = 20e3
        self.dyt[...] = 20e3
        self.dzt[...] = 100.0

    @veros.veros_method
    def set_topography(self):
        self.kbot[:, 2:-2] = 1

    @veros.veros_method
    def set_coriolis(self):
        self.coriolis_t[:,:] = 1e-4

    @veros.veros_method
    def set_initial_conditions(self):
        u_0 = 0.5
        n_0 = 0.004
        f = self.coriolis_t[0, self.ny // 2]
        h = (self.nz - 2) * self.dzt[0]
        kx = 1.6 * f / (n_0 * h)
        ky = self.pi / ((self.ny - 2) * self.dxt[0])
        d = f / n_0 / (kx**2 + ky**2)**0.5

        fxa = 1. / np.tanh(h / d)
        c1 = (1. + 0.25 * (h / d)**2 - h / d * fxa) * complex(1,0)
        c1 = (np.sqrt(c1) * d / h + 0.5) * u_0
        A = (u_0 - c1) / u_0 * h / d

        alpha = veros.core.density.linear_eq.linear_eq_of_state_drhodT()

        # zonal velocity
        self.u[:, :, :, self.tau] = u_0 * (0.5 + self.zt[np.newaxis, np.newaxis, :] / (self.nz * self.dzt[0])) * self.maskU
        self.u[..., self.taum1] = self.u[..., self.tau]

        t0 = -n_0 ** 2 * self.zt[np.newaxis, :] / self.grav / alpha * self.rho_0 * self.maskT[:, 2, :]
        uz = u_0 / self.ht[:, 2:-2, np.newaxis]
        self.t0 = np.zeros((self.nx+4, self.ny+4, self.nz), dtype=self.default_float_type)
        self.t0[:, 3:-1, :] = t0[:, np.newaxis, :] + np.cumsum(self.dyt[np.newaxis, 2:-2, np.newaxis] * uz * f / self.grav / alpha * self.rho_0 * self.maskT[:, 2:-2, :], axis=1)
        self.dt0 = np.zeros((self.nx+4, self.ny+4, self.nz, 3), dtype=self.default_float_type)

        # perturbation buoyancy
        phiz = A / d * np.sinh(self.zt / d) + np.cosh(self.zt / d) / d
        self.temp[..., self.tau] = 0.1 * np.sin(kx * self.xt[:, np.newaxis, np.newaxis]) \
                                    * np.sin(ky * self.yt[np.newaxis, :, np.newaxis]) \
                                    * np.abs(phiz) * self.maskT * self.rho_0 / self.grav / alpha
        self.temp[..., self.taum1] = self.temp[..., self.tau]

        self.t_tot = np.zeros((self.nx+4, self.ny+4, self.nz), dtype=self.default_float_type)

    @veros.veros_method
    def set_forcing(self):
        # update density, etc of last time step
        self.temp[:,:,:,self.tau] = self.temp[:,:,:,self.tau] + self.t0
        veros.core.thermodynamics.calc_eq_of_state(self, self.tau)
        self.t_tot[...] = self.temp[..., self.tau]
        self.temp[:,:,:,self.tau] = self.temp[:,:,:,self.tau] - self.t0

        # advection of background temperature
        veros.core.thermodynamics.advect_tracer(self, self.t0, self.dt0[..., self.tau])
        self.temp_source[:] = (1.5 + self.AB_eps) * self.dt0[..., self.tau] - (0.5 + self.AB_eps) * self.dt0[...,self.taum1]

    @veros.veros_method
    def set_diagnostics(self):
        self.diagnostics["snapshot"].output_frequency = 86400.
        # add total temperature to output
        self.diagnostics["snapshot"].output_variables += ["t_tot"]
        self.variables["t_tot"] = veros.variables.Variable("Total temperature", ("xt", "yt", "zt"), "deg C",
                                                           "Total temperature", output=True, time_dependent=True,
                                                           write_to_restart=True)

    def after_timestep(self):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = Eady(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
