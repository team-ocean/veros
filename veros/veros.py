
import abc

from loguru import logger

from veros import (
    settings, diagnostics, time, handlers, logs, distributed, progress,
    runtime_settings as rs, runtime_state as rst
)
from veros.state import VerosState
from veros.timer import Timer
from veros.core import (
    momentum, numerics, thermodynamics, eke, tke, idemix,
    isoneutral, streamfunction, advection, utilities
)


class VerosSetup(metaclass=abc.ABCMeta):
    """Main class for Veros, used for building a model and running it.

    Note:
        This class is meant to be subclassed. Subclasses need to implement the
        methods :meth:`set_parameter`, :meth:`set_topography`, :meth:`set_grid`,
        :meth:`set_coriolis`, :meth:`set_initial_conditions`, :meth:`set_forcing`,
        and :meth:`set_diagnostics`.

    Arguments:
        backend (:obj:`bool`, optional): Backend to use for array operations.
            Possible values are ``numpy`` and ``bohrium``. Defaults to ``None``, which
            tries to read the backend from the command line (set via a flag
            ``-b``/``--backend``), and uses ``numpy`` if no command line argument is given.
        loglevel (one of {debug, info, warning, error, critical}, optional): Verbosity
            of the model. Tries to read value from command line if not given
            (``-v``/``--loglevel``). Defaults to ``info``.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from veros import VerosSetup
        >>>
        >>> class MyModel(VerosSetup):
        >>>     ...
        >>>
        >>> simulation = MyModel(backend='bohrium')
        >>> simulation.run()
        >>> plt.imshow(simulation.state.psi[..., 0])
        >>> plt.show()

    """

    def __init__(self, state=None, override=None):
        self.override_settings = override or {}
        logs.setup_logging(loglevel=rs.loglevel)

        if state is None:
            self.state = VerosState()

        self.state.timers = {k: Timer(k) for k in (
            'setup', 'main', 'momentum', 'temperature', 'eke', 'idemix',
            'tke', 'diagnostics', 'pressure', 'friction', 'isoneutral',
            'vmix', 'eq_of_state'
        )}

    @abc.abstractmethod
    def set_parameter(self, vs):
        """To be implemented by subclass.

        First function to be called during setup.
        Use this to modify the model settings.

        Example:
          >>> def set_parameter(self, vs):
          >>>     vs.nx, vs.ny, vs.nz = (360, 120, 50)
          >>>     vs.coord_degree = True
          >>>     vs.enable_cyclic = True
        """
        pass

    @abc.abstractmethod
    def set_initial_conditions(self, vs):
        """To be implemented by subclass.

        May be used to set initial conditions.

        Example:
          >>> @veros_method
          >>> def set_initial_conditions(self, vs):
          >>>     vs.u[:, :, :, vs.tau] = np.random.rand(vs.u.shape[:-1])
        """
        pass

    @abc.abstractmethod
    def set_grid(self, vs):
        """To be implemented by subclass.

        Has to set the grid spacings :attr:`dxt`, :attr:`dyt`, and :attr:`dzt`,
        along with the coordinates of the grid origin, :attr:`x_origin` and
        :attr:`y_origin`.

        Example:
          >>> @veros_method
          >>> def set_grid(self, vs):
          >>>     vs.x_origin, vs.y_origin = 0, 0
          >>>     vs.dxt[...] = [0.1, 0.05, 0.025, 0.025, 0.05, 0.1]
          >>>     vs.dyt[...] = 1.
          >>>     vs.dzt[...] = [10, 10, 20, 50, 100, 200]
        """
        pass

    @abc.abstractmethod
    def set_coriolis(self, vs):
        """To be implemented by subclass.

        Has to set the Coriolis parameter :attr:`coriolis_t` at T grid cells.

        Example:
          >>> @veros_method
          >>> def set_coriolis(self, vs):
          >>>     vs.coriolis_t[:, :] = 2 * vs.omega * np.sin(vs.yt[np.newaxis, :] / 180. * vs.pi)
        """
        pass

    @abc.abstractmethod
    def set_topography(self, vs):
        """To be implemented by subclass.

        Must specify the model topography by setting :attr:`kbot`.

        Example:
          >>> @veros_method
          >>> def set_topography(self, vs):
          >>>     vs.kbot[:, :] = 10
          >>>     # add a rectangular island somewhere inside the domain
          >>>     vs.kbot[10:20, 10:20] = 0
        """
        pass

    @abc.abstractmethod
    def set_forcing(self, vs):
        """To be implemented by subclass.

        Called before every time step to update the external forcing, e.g. through
        :attr:`forc_temp_surface`, :attr:`forc_salt_surface`, :attr:`surface_taux`,
        :attr:`surface_tauy`, :attr:`forc_tke_surface`, :attr:`temp_source`, or
        :attr:`salt_source`. Use this method to implement time-dependent forcing.

        Example:
          >>> @veros_method
          >>> def set_forcing(self, vs):
          >>>     current_month = (vs.time / (31 * 24 * 60 * 60)) % 12
          >>>     vs.surface_taux[:, :] = vs._windstress_data[:, :, current_month]
        """
        pass

    @abc.abstractmethod
    def set_diagnostics(self, vs):
        """To be implemented by subclass.

        Called before setting up the :ref:`diagnostics <diagnostics>`. Use this method e.g. to
        mark additional :ref:`variables <variables>` for output.

        Example:
          >>> @veros_method
          >>> def set_diagnostics(self, vs):
          >>>     vs.diagnostics['snapshot'].output_vars += ['drho', 'dsalt', 'dtemp']
        """
        pass

    @abc.abstractmethod
    def after_timestep(self, vs):
        """Called at the end of each time step. Can be used to define custom, setup-specific
        events.
        """
        pass

    def setup(self):
        vs = self.state

        with vs.timers['setup']:
            logger.info('Setting up everything')

            self.set_parameter(vs)

            for setting, value in self.override_settings.items():
                setattr(vs, setting, value)

            settings.check_setting_conflicts(vs)
            distributed.validate_decomposition(vs)
            vs.allocate_variables()

            self.set_grid(vs)
            numerics.calc_grid(vs)

            self.set_coriolis(vs)
            numerics.calc_beta(vs)

            self.set_topography(vs)
            numerics.calc_topo(vs)

            self.set_initial_conditions(vs)
            numerics.calc_initial_conditions(vs)
            streamfunction.streamfunction_init(vs)
            eke.init_eke(vs)

            vs.diagnostics = diagnostics.create_diagnostics(vs)
            self.set_diagnostics(vs)
            diagnostics.initialize(vs)
            diagnostics.read_restart(vs)

            self.set_forcing(vs)
            isoneutral.check_isoneutral_slope_crit(vs)

    def run(self, show_progress_bar=None):
        """Main routine of the simulation.

        Note:
            Make sure to call :meth:`setup` prior to this function.

        Arguments:
            show_progress_bar (:obj:`bool`, optional): Whether to show fancy progress bar via tqdm.
                By default, only show if stdout is a terminal and Veros is running on a single process.

        """
        vs = self.state

        logger.info('\nStarting integration for {0[0]:.1f} {0[1]}'.format(time.format_time(vs.runlen)))

        start_time, start_iteration = vs.time, vs.itt
        profiler = None

        pbar = progress.get_progress_bar(vs, use_tqdm=show_progress_bar)

        with handlers.signals_to_exception():
            try:
                with pbar:
                    while vs.time - start_time < vs.runlen:
                        with vs.timers['diagnostics']:
                            diagnostics.write_restart(vs)

                        if vs.itt - start_iteration == 3 and rs.profile_mode and rst.proc_rank == 0:
                            # when using bohrium, most kernels should be pre-compiled by now
                            profiler = diagnostics.start_profiler()

                        with vs.timers['main']:
                            self.set_forcing(vs)

                            if vs.enable_idemix:
                                idemix.set_idemix_parameter(vs)

                            with vs.timers['eke']:
                                eke.set_eke_diffusivities(vs)

                            with vs.timers['tke']:
                                tke.set_tke_diffusivities(vs)

                            with vs.timers['momentum']:
                                momentum.momentum(vs)

                            with vs.timers['temperature']:
                                thermodynamics.thermodynamics(vs)

                            if vs.enable_eke or vs.enable_tke or vs.enable_idemix:
                                advection.calculate_velocity_on_wgrid(vs)

                            with vs.timers['eke']:
                                if vs.enable_eke:
                                    eke.integrate_eke(vs)

                            with vs.timers['idemix']:
                                if vs.enable_idemix:
                                    idemix.integrate_idemix(vs)

                            with vs.timers['tke']:
                                if vs.enable_tke:
                                    tke.integrate_tke(vs)

                            utilities.enforce_boundaries(vs, vs.u[:, :, :, vs.taup1])
                            utilities.enforce_boundaries(vs, vs.v[:, :, :, vs.taup1])
                            if vs.enable_tke:
                                utilities.enforce_boundaries(vs, vs.tke[:, :, :, vs.taup1])
                            if vs.enable_eke:
                                utilities.enforce_boundaries(vs, vs.eke[:, :, :, vs.taup1])
                            if vs.enable_idemix:
                                utilities.enforce_boundaries(vs, vs.E_iw[:, :, :, vs.taup1])

                            momentum.vertical_velocity(vs)

                        vs.itt += 1
                        vs.time += vs.dt_tracer
                        pbar.advance_time(vs.dt_tracer)

                        self.after_timestep(vs)

                        with vs.timers['diagnostics']:
                            if not diagnostics.sanity_check(vs):
                                raise RuntimeError('solution diverged at iteration {}'.format(vs.itt))

                            if vs.enable_neutral_diffusion and vs.enable_skew_diffusion:
                                isoneutral.isoneutral_diag_streamfunction(vs)

                            diagnostics.diagnose(vs)
                            diagnostics.output(vs)

                        # NOTE: benchmarks parse this, do not change / remove
                        logger.debug(' Time step took {:.2f}s', vs.timers['main'].get_last_time())

                        # permutate time indices
                        vs.taum1, vs.tau, vs.taup1 = vs.tau, vs.taup1, vs.taum1

            except:
                logger.critical('Stopping integration at iteration {}', vs.itt)
                raise

            else:
                logger.success('Integration done\n')

            finally:
                diagnostics.write_restart(vs, force=True)

                logger.debug('\n'.join([
                    '',
                    'Timing summary:',
                    ' setup time               = {:.2f}s'.format(vs.timers['setup'].get_time()),
                    ' main loop time           = {:.2f}s'.format(vs.timers['main'].get_time()),
                    '   momentum               = {:.2f}s'.format(vs.timers['momentum'].get_time()),
                    '     pressure             = {:.2f}s'.format(vs.timers['pressure'].get_time()),
                    '     friction             = {:.2f}s'.format(vs.timers['friction'].get_time()),
                    '   thermodynamics         = {:.2f}s'.format(vs.timers['temperature'].get_time()),
                    '     lateral mixing       = {:.2f}s'.format(vs.timers['isoneutral'].get_time()),
                    '     vertical mixing      = {:.2f}s'.format(vs.timers['vmix'].get_time()),
                    '     equation of state    = {:.2f}s'.format(vs.timers['eq_of_state'].get_time()),
                    '   EKE                    = {:.2f}s'.format(vs.timers['eke'].get_time()),
                    '   IDEMIX                 = {:.2f}s'.format(vs.timers['idemix'].get_time()),
                    '   TKE                    = {:.2f}s'.format(vs.timers['tke'].get_time()),
                    ' diagnostics and I/O      = {:.2f}s'.format(vs.timers['diagnostics'].get_time()),
                ]))

                if profiler is not None:
                    diagnostics.stop_profiler(profiler)
