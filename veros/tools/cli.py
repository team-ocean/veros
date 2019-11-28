import functools
import sys
import time

import click

from veros.settings import SETTINGS

BACKENDS = ['numpy', 'bohrium']
LOGLEVELS = ['trace', 'debug', 'info', 'warning', 'error', 'critical']


class VerosSetting(click.ParamType):
    name = 'setting'
    current_key = None

    def convert(self, value, param, ctx):
        assert param.nargs == 2

        if self.current_key is None:
            if value not in SETTINGS:
                self.fail('Unknown setting %s' % value)
            self.current_key = value
            return value

        assert self.current_key in SETTINGS
        setting = SETTINGS[self.current_key]
        self.current_key = None

        if setting.type is bool:
            return click.BOOL(value)

        return setting.type(value)


def cli(run):
    """Decorator that wraps the decorated function with the Veros setup command line interface.

    Example:

        >>> @veros.tools.cli.cli()
        >>> def run_setup(override):
        ...     sim = MyVerosSetup(override=override)
        ...     sim.run()
        ...
        >>> if __name__ == '__main__':
        ...     run_setup()

    This script then automatically supports settings to be specified from the command line::

        $ python my_setup.py --help
        Usage: my_setup.py [OPTIONS]

        Options:
        -b, --backend [numpy|bohrium]   Backend to use for computations (default:
                                        numpy)
        -v, --loglevel [trace|debug|info|warning|error|critical]
                                        Log level used for output (default: info)
        -s, --override SETTING VALUE    Override default setting, may be specified
                                        multiple times
        -p, --profile-mode              Write a performance profile for debugging
                                        (default: false)
        -n, --num-proc INTEGER...       Number of processes in x and y dimension
                                        (requires execution via mpirun)
        --help                          Show this message and exit.

    """
    @click.command('veros-run')
    @click.option('-b', '--backend', default='numpy', type=click.Choice(BACKENDS),
                  help='Backend to use for computations (default: numpy)', envvar='VEROS_BACKEND')
    @click.option('-v', '--loglevel', default='info', type=click.Choice(LOGLEVELS),
                  help='Log level used for output (default: info)', envvar='VEROS_LOGLEVEL')
    @click.option('-s', '--override', nargs=2, multiple=True, metavar='SETTING VALUE',
                  type=VerosSetting(), default=tuple(),
                  help='Override default setting, may be specified multiple times')
    @click.option('-p', '--profile-mode', is_flag=True, default=False, type=click.BOOL, envvar='VEROS_PROFILE',
                  help='Write a performance profile for debugging (default: false)')
    @click.option('-n', '--num-proc', nargs=2, default=[1, 1], type=click.INT,
                  help='Number of processes in x and y dimension')
    @click.option('--slave', default=False, is_flag=True, hidden=True,
                  help='Indicates that this process is an MPI worker (for internal use)')
    @functools.wraps(run)
    def wrapped(*args, slave, **kwargs):
        from veros import runtime_settings, runtime_state

        total_proc = kwargs['num_proc'][0] * kwargs['num_proc'][1]

        if total_proc > 1 and runtime_state.proc_num == 1 and not slave:
            from mpi4py import MPI

            comm = MPI.COMM_SELF.Spawn(
                sys.executable,
                args=['-m', 'mpi4py'] + list(sys.argv) + ['--slave'],
                maxprocs=total_proc
            )

            futures = [comm.irecv(source=p) for p in range(total_proc)]
            while True:
                done, success = zip(*(f.test() for f in futures))

                if any(s is False for s in success):
                    raise RuntimeError('An MPI worker encountered an error')

                if all(done):
                    break

                time.sleep(0.1)

            return

        kwargs['override'] = dict(kwargs['override'])

        for setting in ('backend', 'profile_mode', 'num_proc', 'loglevel'):
            setattr(runtime_settings, setting, kwargs.pop(setting))

        try:
            run(*args, **kwargs)
        except:  # noqa: E722
            status = False
            raise
        else:
            status = True
        finally:
            if slave:
                runtime_settings.mpi_comm.Get_parent().send(status, dest=0)

    return wrapped
