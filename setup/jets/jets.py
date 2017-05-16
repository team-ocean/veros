import math

from veros import Veros, veros_method, core

HRESOLVE = 0.5
VRESOLVE = 0.5
N_0 = 0.004
M_0 = math.sqrt(1e-5 * 0.1 / 1024. * 9.801)
spg_width = int(3 * HRESOLVE)
t_rest = 1. / (5. * 86400)

DX = 30e3
Lx = DX * 128
H = 1800.0

class Jets(Veros):
    """An idealized configuration to demonstrate eddy-driven zonal jets.

    A wide channel model with relaxation at the side walls and interior damping as in
    Eden (2011) simulating strong eddy-driven zonal jets.

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/zonal%20jets>`_.
    """
    @veros_method
    def set_parameter(self):
        self.identifier = "jets"

        self.nx, self.ny, self.nz = int(128 * HRESOLVE), int(128 * HRESOLVE), int(18 * VRESOLVE)
        self.dt_tracer = 1800. / HRESOLVE
        self.dt_mom = 1800. / HRESOLVE
        self.runlen = 365 * 86400.

        self.enable_conserve_energy = False
        self.coord_degree = False
        self.enable_cyclic_x = True
        self.eq_of_state_type = 1

        self.congr_epsilon = 1e-12
        self.congr_max_iterations = 5000

        self.kappaH_0 = 1e-4 / VRESOLVE ** 2
        self.kappaM_0 = 1e-4 / VRESOLVE ** 2

        self.enable_ray_friction = True
        self.r_ray = 1e-7

        #self.enable_hor_friction = 1
        #self.a_h = 100/HRESOLVE**2
        self.enable_biharmonic_friction = True
        self.A_hbi = 5e11 / HRESOLVE ** 2

        self.enable_superbee_advection = True
        self.enable_tempsalt_sources = True

    @veros_method
    def set_grid(self):
        self.x_origin, self.y_origin = 0., 0.
        self.dxt[:] = Lx / self.nx
        self.dyt[:] = Lx / self.ny
        self.dzt[:] = H / self.nz

    @veros_method
    def set_coriolis(self):
        phi0 = 10. / 180. * self.pi
        betaloc = 2 * self.omega * np.cos(phi0) / self.radius
        self.coriolis_t[:,:] = 2 * self.omega * np.sin(phi0) + betaloc * self.yt[np.newaxis, :]

    @veros_method
    def set_topography(self):
        self.kbot[:] = 1

    @veros_method
    def set_initial_conditions(self):
        alpha = core.density.linear_eq.linear_eq_of_state_drhodT()
        self.t_star = np.zeros((self.nx+4, self.ny+4, self.nz))
        self.t_rest = np.zeros((self.nx+4, self.ny+4, self.nz))

        fxa = 0.5e-3 * np.sin(self.xt * 8.5 / Lx * self.pi) * 1024. / 9.81 / alpha
        self.t_star[...] = (32 + (M_0**2 * self.yt[np.newaxis, :, np.newaxis] \
                                - N_0**2 * self.zt[np.newaxis, np.newaxis, :]) \
                            * 1024. / 9.81 / alpha) * self.maskT
        self.temp[..., self.tau] = (fxa[:, np.newaxis, np.newaxis] + self.t_star) * self.maskT
        self.temp[..., self.taum1] = self.temp[... ,self.tau]

        self.t_rest[:, 2:spg_width+2, :] = t_rest * self.maskT[:, 2:spg_width+2, :] \
                                           / np.arange(1, spg_width+1)[np.newaxis, :, np.newaxis]
        self.t_rest[:, -spg_width-2:-2, :] = t_rest * self.maskT[:, -spg_width-2:-2, :] \
                                           / np.arange(spg_width, 0, -1)[np.newaxis, :, np.newaxis]

    @veros_method
    def set_forcing(self):
        if self.enable_tempsalt_sources:
            self.temp_source[:] = self.t_rest * (self.t_star - self.temp[:,:,:,self.tau]) * self.maskT

    @veros_method
    def set_diagnostics(self):
        self.diagnostics["snapshot"].output_frequency = 3 * 86400.


if __name__ == "__main__":
    simulation = Jets()
    simulation.setup()
    simulation.run()
