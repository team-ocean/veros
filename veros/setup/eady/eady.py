from veros import VerosSetup, veros_method
import veros.tools


class EadySetup(VerosSetup):
    """Demonstrates the classical EadySetup (1949) solution.

    A narrow channel on an f-plane with prescribed stratification and vertically sheared background zonal flow.
    The temperature variable of the model is in this case a temperature perturbation,
    the effect of the background stratification on the perturbation temperature is implemented in
    the configuration routine set_forcing.

    This setup demonstrates:
     - setting up a highly idealized configuration
     - adding additional variables to the model
     - adding additional variables to snapshot output
     - calling core routines from the setup

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/EadySetup%201>`_.
    """
    @veros_method
    def set_parameter(self, vs):
        vs.identifier = "eady"

        vs.nx, vs.ny, vs.nz = 32, 32, 20
        vs.dt_tracer = 1200.
        vs.dt_mom = 1200.
        vs.runlen = 100 * 86400.

        vs.congr_epsilon = 1e-12
        vs.congr_max_iterations = 5000

        vs.enable_cyclic_x = True

        vs.enable_superbee_advection = True
        vs.enable_explicit_vert_friction = True
        vs.enable_hor_friction = True
        vs.A_h = (20e3)**3 * 2e-11
        vs.kappaM_0 = 1.e-4
        vs.kappaH_0 = 1.e-4

        vs.enable_conserve_energy = False
        vs.coord_degree = False
        vs.eq_of_state_type = 1
        vs.enable_tempsalt_sources = True

    @veros_method
    def set_grid(self, vs):
        vs.x_origin = 0.
        vs.y_origin = 0.
        vs.dxt[...] = 20e3
        vs.dyt[...] = 20e3
        vs.dzt[...] = 100.0

    @veros_method
    def set_topography(self, vs):
        vs.kbot[:, 2:-2] = 1

    @veros_method
    def set_coriolis(self, vs):
        vs.coriolis_t[:, :] = 1e-4

    @veros_method
    def set_initial_conditions(self, vs):
        u_0 = 0.5
        n_0 = 0.004
        f = vs.coriolis_t[0, vs.ny // 2]
        h = (vs.nz - 2) * vs.dzt[0]
        kx = 1.6 * f / (n_0 * h)
        ky = vs.pi / ((vs.ny - 2) * vs.dxt[0])
        d = f / n_0 / (kx**2 + ky**2)**0.5

        fxa = 1. / np.tanh(h / d)
        c1 = (1. + 0.25 * (h / d)**2 - h / d * fxa) * complex(1,0)
        c1 = (np.sqrt(c1) * d / h + 0.5) * u_0
        A = (u_0 - c1) / u_0 * h / d

        alpha = veros.core.density.linear_eq.linear_eq_of_state_drhodT()

        # zonal velocity
        vs.u[:, :, :, vs.tau] = u_0 * (0.5 + vs.zt[np.newaxis, np.newaxis, :] / (vs.nz * vs.dzt[0])) * vs.maskU
        vs.u[..., vs.taum1] = vs.u[..., vs.tau]

        t0 = -n_0 ** 2 * vs.zt[np.newaxis, :] / vs.grav / alpha * vs.rho_0 * vs.maskT[:, 2, :]
        uz = u_0 / vs.ht[:, 2:-2, np.newaxis]
        vs.t0 = np.zeros((vs.nx+4, vs.ny+4, vs.nz), dtype=vs.default_float_type)
        vs.t0[:, 3:-1, :] = t0[:, np.newaxis, :] + np.cumsum(vs.dyt[np.newaxis, 2:-2, np.newaxis] * uz * f / vs.grav / alpha * vs.rho_0 * vs.maskT[:, 2:-2, :], axis=1)
        vs.dt0 = np.zeros((vs.nx+4, vs.ny+4, vs.nz, 3), dtype=vs.default_float_type)

        # perturbation buoyancy
        phiz = A / d * np.sinh(vs.zt / d) + np.cosh(vs.zt / d) / d
        vs.temp[..., vs.tau] = 0.1 * np.sin(kx * vs.xt[:, np.newaxis, np.newaxis]) \
                                    * np.sin(ky * vs.yt[np.newaxis, :, np.newaxis]) \
                                    * np.abs(phiz) * vs.maskT * vs.rho_0 / vs.grav / alpha
        vs.temp[..., vs.taum1] = vs.temp[..., vs.tau]

        vs.t_tot = np.zeros((vs.nx+4, vs.ny+4, vs.nz), dtype=vs.default_float_type)

    @veros_method
    def set_forcing(self, vs):
        # update density, etc of last time step
        vs.temp[:,:,:,vs.tau] = vs.temp[:,:,:,vs.tau] + vs.t0
        veros.core.thermodynamics.calc_eq_of_state(vs, vs.tau)
        vs.t_tot[...] = vs.temp[..., vs.tau]
        vs.temp[:,:,:,vs.tau] = vs.temp[:,:,:,vs.tau] - vs.t0

        # advection of background temperature
        veros.core.thermodynamics.advect_tracer(vs, vs.t0, vs.dt0[..., vs.tau])
        vs.temp_source[:] = (1.5 + vs.AB_eps) * vs.dt0[..., vs.tau] - (0.5 + vs.AB_eps) * vs.dt0[...,vs.taum1]

    @veros_method
    def set_diagnostics(self, vs):
        vs.diagnostics["snapshot"].output_frequency = 86400.
        # add total temperature to output
        vs.diagnostics["snapshot"].output_variables += ["t_tot"]
        vs.variables["t_tot"] = veros.variables.Variable("Total temperature", ("xt", "yt", "zt"), "deg C",
                                                         "Total temperature", output=True, time_dependent=True,
                                                         write_to_restart=True)

    def after_timestep(self, vs):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = EadySetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
