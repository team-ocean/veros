import logging

from .. import pyom_method

def initialize(pyom):
    pass

@pyom_method
def diagnose(pyom):
    """
    Diagnose tracer content
    """
    cell_volume = pyom.area_t[2:-2, 2:-2, np.newaxis] * pyom.dzt[np.newaxis, np.newaxis, :] \
                  * pyom.maskT[2:-2, 2:-2, :]
    volm = np.sum(cell_volume)
    tempm = np.sum(cell_volume * pyom.temp[2:-2, 2:-2, :, pyom.tau])
    saltm = np.sum(cell_volume * pyom.salt[2:-2, 2:-2, :, pyom.tau])
    vtemp = np.sum(cell_volume * pyom.temp[2:-2, 2:-2, :, pyom.tau]**2)
    vsalt = np.sum(cell_volume * pyom.salt[2:-2, 2:-2, :, pyom.tau]**2)

    logging.warning("")
    logging.warning("mean temperature {} change to last {}".format(tempm/volm, (tempm-diag_tracer_content.tempm1)/volm))
    logging.warning("mean salinity    {} change to last {}".format(saltm/volm, (saltm-diag_tracer_content.saltm1)/volm))
    logging.warning("temperature var. {} change to last {}".format(vtemp/volm, (vtemp-diag_tracer_content.vtemp1)/volm))
    logging.warning("salinity var.    {} change to last {}".format(vsalt/volm, (vsalt-diag_tracer_content.vsalt1)/volm))

    diagnose.tempm1 = tempm
    diagnose.vtemp1 = vtemp
    diagnose.saltm1 = saltm
    diagnose.vsalt1 = vsalt
diagnose.tempm1 = 0.
diagnose.saltm1 = 0.
diagnose.vtemp1 = 0.
diagnose.vsalt1 = 0.


def output(pyom):
    pass
