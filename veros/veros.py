import math
import logging

# for some reason, netCDF4 has to be imported before h5py, so we do it here
import netCDF4
import h5py

from . import variables, settings, cli, diagnostics, time, handlers
from . import backend as _backend
from .timer import Timer
from .core import momentum, numerics, thermodynamics, eke, tke, idemix, \
                  isoneutral, external, advection, cyclic


class Veros(object):
    """Main class for Veros, used for building a model and running it.

    Note:
        This class is meant to be subclassed. Subclasses need to implement the
        methods :meth:`set_parameter`, :meth:`set_topography`, :meth:`set_grid`,
        :meth:`set_coriolis`, :meth:`set_initial_conditions`, :meth:`set_forcing`,
        and :meth:`set_diagnostics`.

    Args:
        backend (:obj:`bool`, optional): Backend to use for array operations.
            Possible values are ``numpy`` and ``bohrium``. Defaults to ``None``, which
            tries to read the backend from the command line (set via a flag
            ``-b``/``--backend``), and uses ``numpy`` if no command line argument is given.
        loglevel (one of {debug, info, warning, error, critical}, optional): Verbosity
            of the model. Tries to read value from command line if not given
            (``-v``/``--loglevel``). Defaults to ``info``.
        logfile (path, optional): Path to a log file to write output to. Tries to
            read value from command line if not given (``-l``/``--logfile``). Defaults
            to stdout.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from climate.veros import Veros
        >>>
        >>> class MyModel(Veros):
        >>>     ...
        >>>
        >>> simulation = MyModel(backend="bohrium")
        >>> simulation.run()
        >>> plt.imshow(simulation.psi)
        >>> plt.show()
    """

    # Constants
    pi = math.pi
    radius = 6370e3  # Earth radius in m
    degtom = radius / 180. * pi  # Conversion degrees latitude to meters
    mtodeg = 1. / degtom  # Conversion meters to degrees latitude
    omega = pi / 43082.  # Earth rotation frequency in 1/s
    rho_0 = 1024.  # Boussinesq reference density in :math:`kg/m^3`
    grav = 9.81  # Gravitational constant in :math:`m/s^2`

    def __init__(self, backend=None, loglevel=None, logfile=None):
        args = cli.parse_command_line()
        self.command_line_settings = args.set or {}
        self.profile_mode = args.profile
        self.backend, self.backend_name = _backend.get_backend(backend or args.backend)

        try: # python 2
            logging.basicConfig(logfile=logfile or args.logfile, filemode="w",
                                level=getattr(logging, (loglevel or args.loglevel).upper()),
                                format="%(message)s")
        except ValueError: # python 3
            logging.basicConfig(filename=logfile or args.logfile, filemode="w",
                                level=getattr(logging, (loglevel or args.loglevel).upper()),
                                format="%(message)s")

        settings.set_default_settings(self)

        self.timers = {k: Timer(k) for k in
            (
                "setup", "main", "momentum", "temperature", "eke", "idemix",
                "tke", "diagnostics", "pressure", "friction", "isoneutral",
                "vmix", "eq_of_state"
            )}

        self.poisson_solver = None
        self.nisle = 0 # to be overriden during streamfunction_init
        self.taum1, self.tau, self.taup1 = 0, 1, 2 # pointers to last, current, and next time step
        self.time, self.itt = 0., 1 # current time and iteration

    def _not_implemented(self):
        raise NotImplementedError("Needs to be implemented by subclass")

    def set_parameter(self):
        """To be implemented by subclass.

        First function to be called during setup.
        Use this to modify the model settings.

        Example:
          >>> def set_parameter(self):
          >>>     self.nx, self.ny, self.nz = (360, 120, 50)
          >>>     self.coord_degree = True
          >>>     self.enable_cyclic = True
        """
        self._not_implemented()

    def set_initial_conditions(self):
        """To be implemented by subclass.

        May be used to set initial conditions.

        Example:
          >>> @veros_method
          >>> def set_initial_conditions(self):
          >>>     self.u[:, :, :, self.tau] = np.random.rand(self.u.shape[:-1])
        """
        self._not_implemented()

    def set_grid(self):
        """To be implemented by subclass.

        Has to set the grid spacings :attr:`dxt`, :attr:`dyt`, and :attr:`dzt`,
        along with the coordinates of the grid origin, :attr:`x_origin` and
        :attr:`y_origin`.

        Example:
          >>> @veros_method
          >>> def set_grid(self):
          >>>     self.x_origin, self.y_origin = 0, 0
          >>>     self.dxt[...] = [0.1, 0.05, 0.025, 0.025, 0.05, 0.1]
          >>>     self.dyt[...] = 1.
          >>>     self.dzt[...] = [10, 10, 20, 50, 100, 200]
        """
        self._not_implemented()

    def set_coriolis(self):
        """To be implemented by subclass.

        Has to set the Coriolis parameter :attr:`coriolis_t` at T grid cells.

        Example:
          >>> @veros_method
          >>> def set_coriolis(self):
          >>>     self.coriolis_t[:, :] = 2 * self.omega * np.sin(self.yt[np.newaxis, :] / 180. * self.pi)
        """
        self._not_implemented()

    def set_topography(self):
        """To be implemented by subclass.

        Must specify the model topography by setting :attr:`kbot`.

        Example:
          >>> @veros_method
          >>> def set_topography(self):
          >>>     self.kbot[:, :] = 10
          >>>     # add a rectangular island somewhere inside the domain
          >>>     self.kbot[10:20, 10:20] = 0
        """
        self._not_implemented()

    def set_forcing(self):
        """To be implemented by subclass.

        Called before every time step to update the external forcing, e.g. through
        :attr:`forc_temp_surface`, :attr:`forc_salt_surface`, :attr:`surface_taux`,
        :attr:`surface_tauy`, :attr:`forc_tke_surface`, :attr:`temp_source`, or
        :attr:`salt_source`. Use this method to implement time-dependent forcing.

        Example:
          >>> @veros_method
          >>> def set_forcing(self):
          >>>     current_month = (self.time / (31 * 24 * 60 * 60)) % 12
          >>>     self.surface_taux[:, :] = self._windstress_data[:, :, current_month]
        """
        self._not_implemented()

    def set_diagnostics(self):
        """To be implemented by subclass.

        Called before setting up the :ref:`diagnostics <diagnostics>`. Use this method e.g. to
        mark additional :ref:`variables <variables>` for output.

        Example:
          >>> @veros_method
          >>> def set_diagnostics(self):
          >>>     self.diagnostics["snapshot"].output_vars += ["drho", "dsalt", "dtemp"]
        """
        self._not_implemented()

    def after_timestep(self):
        """Called at the end of each time step. Can be used to define custom, setup-specific
        events.
        """
        pass

    def flush(self):
        """Flush computations if supported by the current backend.
        """
        try:
            self.backend.flush()
        except AttributeError:
            pass

    def setup(self):
        with self.timers["setup"]:
            logging.info("Setting up everything")

            self.set_parameter()
            cli.set_commandline_settings(self)
            settings.check_setting_conflicts(self)
            variables.allocate_variables(self)

            self.set_grid()
            numerics.calc_grid(self)

            self.set_coriolis()
            numerics.calc_beta(self)

            self.set_topography()
            numerics.calc_topo(self)

            self.set_initial_conditions()
            numerics.calc_initial_conditions(self)

            self.set_forcing()
            external.streamfunction_init(self)

            logging.info("Initializing diagnostics")
            self.diagnostics = diagnostics.create_diagnostics(self)
            self.set_diagnostics()
            diagnostics.initialize(self)

            eke.init_eke(self)

            isoneutral.check_isoneutral_slope_crit(self)

            diagnostics.read_restart(self)


    def run(self):
        """Main routine of the simulation.
        """
        enditt = self.itt + int(self.runlen / self.dt_tracer) - 1
        logging.info("Starting integration for {0[0]:.1f} {0[1]}".format(time.format_time(self, self.runlen)))
        logging.info(" from time step {} to {}".format(self.itt, enditt))

        start_time, start_iteration = self.time, self.itt
        profiler = None
        with handlers.signals_to_exception():
            try:
                while self.time - start_time < self.runlen:
                    logging.info("Current iteration: {}".format(self.itt))

                    with self.timers["diagnostics"]:
                        diagnostics.write_restart(self)

                    if self.itt - start_iteration == 3 and self.profile_mode:
                        # when using bohrium, most kernels should be pre-compiled by now
                        profiler = diagnostics.start_profiler()

                    with self.timers["main"]:
                        self.set_forcing()

                        if self.enable_idemix:
                            idemix.set_idemix_parameter(self)

                        with self.timers["eke"]:
                            eke.set_eke_diffusivities(self)

                        with self.timers["tke"]:
                            tke.set_tke_diffusivities(self)

                        with self.timers["momentum"]:
                            momentum.momentum(self)

                        with self.timers["temperature"]:
                            thermodynamics.thermodynamics(self)

                        if self.enable_eke or self.enable_tke or self.enable_idemix:
                            advection.calculate_velocity_on_wgrid(self)

                        with self.timers["eke"]:
                            if self.enable_eke:
                                eke.integrate_eke(self)

                        with self.timers["idemix"]:
                            if self.enable_idemix:
                                idemix.integrate_idemix(self)

                        with self.timers["tke"]:
                            if self.enable_tke:
                                tke.integrate_tke(self)

                        if self.enable_cyclic_x:
                            cyclic.setcyclic_x(self.u[:, :, :, self.taup1])
                            cyclic.setcyclic_x(self.v[:, :, :, self.taup1])
                            if self.enable_tke:
                                cyclic.setcyclic_x(self.tke[:, :, :, self.taup1])
                            if self.enable_eke:
                                cyclic.setcyclic_x(self.eke[:, :, :, self.taup1])
                            if self.enable_idemix:
                                cyclic.setcyclic_x(self.E_iw[:, :, :, self.taup1])

                        momentum.vertical_velocity(self)

                    self.itt += 1
                    self.time += self.dt_tracer

                    self.after_timestep()

                    with self.timers["diagnostics"]:
                        if not diagnostics.sanity_check(self):
                            raise RuntimeError("solver diverged at iteration {}".format(self.itt))

                        if self.enable_neutral_diffusion and self.enable_skew_diffusion:
                            isoneutral.isoneutral_diag_streamfunction(self)

                        diagnostics.diagnose(self)
                        diagnostics.output(self)

                    logging.debug("Time step took {}s".format(self.timers["main"].getLastTime()))

                    # permutate time indices
                    self.taum1, self.tau, self.taup1 = self.tau, self.taup1, self.taum1

            except:
                logging.critical("stopping integration at iteration {}".format(self.itt))
                raise

            finally:
                diagnostics.write_restart(self, force=True)
                logging.debug("\n".join([
                    "Timing summary:",
                    " setup time               = {:.2f}s".format(self.timers["setup"].getTime()),
                    " main loop time           = {:.2f}s".format(self.timers["main"].getTime()),
                    "     momentum             = {:.2f}s".format(self.timers["momentum"].getTime()),
                    "       pressure           = {:.2f}s".format(self.timers["pressure"].getTime()),
                    "       friction           = {:.2f}s".format(self.timers["friction"].getTime()),
                    "     thermodynamics       = {:.2f}s".format(self.timers["temperature"].getTime()),
                    "       lateral mixing     = {:.2f}s".format(self.timers["isoneutral"].getTime()),
                    "       vertical mixing    = {:.2f}s".format(self.timers["vmix"].getTime()),
                    "       equation of state  = {:.2f}s".format(self.timers["eq_of_state"].getTime()),
                    "     EKE                  = {:.2f}s".format(self.timers["eke"].getTime()),
                    "     IDEMIX               = {:.2f}s".format(self.timers["idemix"].getTime()),
                    "     TKE                  = {:.2f}s".format(self.timers["tke"].getTime()),
                    " diagnostics and I/O      = {:.2f}s".format(self.timers["diagnostics"].getTime()),
                ]))
                if profiler is not None:
                    diagnostics.stop_profiler(profiler)
                logging.shutdown()
