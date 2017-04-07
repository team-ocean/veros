from collections import OrderedDict
import logging

from . import io_tools
from .. import veros_class_method
from .diagnostic import VerosDiagnostic
from ..core import density
from ..variables import Variable

OVERTURNING_VARIABLES = OrderedDict([
    ("sigma", Variable(
        "Sigma axis", ("sigma",), "kg/m^3", "Sigma axis", output=True
    )),
    ("trans", Variable(
        "Meridional transport", ("yu", "sigma"), "m^3/s",
        "Meridional transport", time_dependent=True, output=True
    )),
    ("vsf_iso", Variable(
        "Meridional transport", ("yu", "zw"), "m^3/s",
        "Meridional transport", time_dependent=True, output=True
    )),
    ("vsf_depth", Variable(
        "Meridional transport", ("yu", "zw"), "m^3/s",
        "Meridional transport", time_dependent=True, output=True
    )),
])
ISONEUTRAL_VARIABLES = OrderedDict([
    ("bolus_iso", Variable(
        "Meridional transport", ("yu", "zw"), "m^3/s",
        "Meridional transport", time_dependent=True, output=True
    )),
    ("bolus_depth", Variable(
        "Meridional transport", ("yu", "zw"), "m^3/s",
        "Meridional transport", time_dependent=True, output=True
    )),
])

class Overturning(VerosDiagnostic):
    """isopycnal overturning diagnostic
    """
    outfile="{identifier}_overturning.nc"
    p_ref = 2000.

    @veros_class_method
    def initialize(self, veros):
        self.filename = self.outfile.format(**vars(veros))

        self.variables = OVERTURNING_VARIABLES
        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            self.variables.update(ISONEUTRAL_VARIABLES)

        self.nitts = 0
        self.nlevel = veros.nz * 4

        self.sigma = np.zeros(self.nlevel)
        self.trans = np.zeros((veros.ny+4, self.nlevel))
        self.zarea = np.zeros((veros.ny+4, veros.nz))
        self.mean_trans = np.zeros((veros.ny+4, self.nlevel))
        self.mean_vsf_iso = np.zeros((veros.ny+4, veros.nz))
        self.mean_vsf_depth = np.zeros((veros.ny+4, veros.nz))
        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            self.mean_bolus_iso = np.zeros((veros.ny+4, veros.nz))
            self.mean_bolus_depth = np.zeros((veros.ny+4, veros.nz))

        # sigma levels
        self.sige = density.get_rho(veros, 35., -2., self.p_ref)
        self.sigs = density.get_rho(veros, 35., 30., self.p_ref)
        self.dsig = (self.sige - self.sigs) / (self.nlevel - 1)

        logging.debug(" sigma ranges for overturning diagnostic:")
        logging.debug(" start sigma0 = {:.1f}".format(self.sigs))
        logging.debug(" end sigma0 = {:.1f}".format(self.sige))
        logging.debug(" Delta sigma0 = {:.1e}".format(self.dsig))
        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            logging.debug(" also calculating overturning by eddy-driven velocities")

        self.sigma[...] = self.sigs + self.dsig * np.arange(self.nlevel)

        # precalculate area below z levels
        self.zarea[2:-2, :] = np.sum(
                                  veros.dxt[2:-2, np.newaxis, np.newaxis] \
                                * veros.cosu[np.newaxis, 2:-2, np.newaxis] \
                                * veros.maskV[2:-2, 2:-2, :]
                             , axis=0)
        self.zarea[:, 1:] = self.zarea[:, :-1] + self.zarea[:, 1:] * veros.dzt[np.newaxis, 1:]

        # prepare cdf file for output
        with io_tools.threaded_io(veros, self.filename, "w") as f:
            io_tools.initialize_file(veros, f)
            io_tools.add_dimension(veros, "sigma", self.nlevel, f)
            for key, var in self.variables.items():
                if var.time_dependent:
                    var_data = None
                else:
                    var_data = getattr(self, key)
                io_tools.initialize_variable(veros, key, var, f, var_data=var_data)

    @veros_class_method
    def _allocate(self, veros):
        pass


    @veros_class_method
    def diagnose(self, veros):

        # sigma at p_ref
        self.sig_loc = density.get_rho(veros,
                                  veros.salt[2:-2, 2:-1, :, veros.tau],
                                  veros.temp[2:-2, 2:-1, :, veros.tau],
                                  self.p_ref)

        # transports below isopycnals and area below isopycnals


    @veros_class_method
    def output(self, veros):
        import warnings
        warnings.warn("routine is not implemented yet")
