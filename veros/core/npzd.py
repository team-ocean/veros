"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
import numpy as np  # NOTE np is already defined somehow
from .. import veros_method
# from . import numerics, utilities

@veros_method
def npzd(vs):
    """
    Integrate biochemistry: phytoplankton, zooplankton, detritus, po4
    """

    # Import and export of detritus in currently evaluated layer
    detritus_export = np.zeros((vs.detritus.shape[:2]))
    detritus_import = np.zeros_like(detritus_export)

    # Integrated phytplankton
    phyto_integrated = np.zeros_like(vs.phytoplankton[:, :, 0])


    # sinking speed of detritus
    # can't sink beyond bottom
    wd = np.empty_like(vs.phytoplankton)
    wd[:, :] = (vs.wd0 + vs.mw * np.where(-vs.zw[::-1] < vs.mwz, -vs.zw[::-1], vs.mwz)) / vs.dzt[::-1]
    wd *= np.logical_not(vs.maskT)  # TODO: Is this the correct way to use the mask?
                                    # The intention is to zero any element not in water
    nbio = vs.dt_mom // vs.dt_bio
    # ztt = depth at top of grid box
    ztt = vs.zw[::-1]

    # copy the incomming light forcing
    # we use this variable to decrease the available light
    # down the water column
    swr = vs.swr.copy()

    for k in range(vs.phytoplankton.shape[2]):
        # print("layer", k)

        # incomming radiation at layer
        swr = swr * np.exp(- vs.light_attenuation_phytoplankton * phyto_integrated)
        phyto_integrated = np.maximum(vs.phytoplankton[:, :, k], vs.trcmin) * vs.dzt[::-1][k]
        grid_light = swr * np.exp(ztt[k] * vs.rctheta)  # light at top of grid box

        # calculate detritus import pr time step from layer above
        detritus_import = detritus_export / vs.dzt[::-1][k] / vs.dt_bio

        # reset export
        detritus_export[:, :] = 0

        # TODO What is this?
        bct = vs.bbio ** (vs.cbio * vs.temp[:, :, k, vs.taum1])
        bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, k, vs.taum1], 20))  # TODO: Remove magic number
        nud = vs.nud0


        """
        --------------------------------------
        Call the npzd ecosystem dynamics model
        """

        # NOTE this was originally in its own subroutine

        po4_in = vs.po4[:, :, k]
        phytoplankton_in = vs.phytoplankton[:, :, k]
        zooplankton_in = vs.zooplankton[:, :, k]
        detritus_in = vs.detritus[:, :, k]

        # flags to prevent more outgoing flux than available capacity - stability?
        po4flag = po4_in > vs.trcmin
        phytoflag = phytoplankton_in > vs.trcmin
        zooflag = zooplankton_in > vs.trcmin
        detrflag = detritus_in > vs.trcmin

        po4_in = np.maximum(po4_in, vs.trcmin)
        phytoplankton_in = np.maximum(phytoplankton_in, vs.trcmin)
        zooplankton_in = np.maximum(zooplankton_in, vs.trcmin)
        detritus_in = np.maximum(detritus_in, vs.trcmin)


        # ----------------------------
        # Fotosynthesis
        # After Evans & Parslow (1985)
        # ----------------------------
        f1 = np.exp((- vs.light_attenuation_water - vs.light_attenuation_phytoplankton * phytoplankton_in) * vs.dzt[::-1][k])

        # TODO what are these things supposed to be?
        jmax = vs.abio_P * bct
        gd = jmax * vs.dt_mom  #/ 84600* vs.dt_mom
        u1 = np.maximum(grid_light / gd, 1e-6)  # FIXME magic number
        u2 = u1 * f1

        # NOTE: There is an approximation here: u1 < 20 WTF? Why 20?
        phi1 = np.log(u1 + np.sqrt(1 + u1**2)) - (np.sqrt(1 + u1) - 1) / u1
        phi2 = np.log(u2 + np.sqrt(1 + u2**2)) - (np.sqrt(1 + u2) - 1) / u2

        # FIXME It is not clear, what this is supposed to be..
        avej = gd * (phi1 - phi2) / ((vs.light_attenuation_water + vs.light_attenuation_phytoplankton * phytoplankton_in) * vs.dzt[::-1][k])

        # -----------------------------
        # Grazing
        # -----------------------------

        # Maximum grazing rate is a function of temperature
        # bctz sets an upper limit on effects of temperature on grazing
        # phytoplankon growth rates bct are unlimited
        # FIXME: These quantities are not well defined or explained...
        gmax = vs.gbio * bctz
        nupt = vs.nupt0 * bct


        for _ in range(nbio):

            # TODO: What do these do? What do the variables mean?
            # TODO saturation / redfield is constant -> calculate only once
            limP = po4_in / (vs.saturation_constant_N / vs.redfield_ratio_PN + po4_in)
            u_P = np.minimum(avej, jmax * limP)

            # Michaelis-Menten denominator
            thetaZ = vs.zprefP * phytoplankton_in + vs.zprefDet * detritus_in + vs.zprefZ * zooplankton_in + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN


            # ingestion of the different substances...?
            ing_P = vs.zprefP / thetaZ
            ing_Det = vs.zprefDet / thetaZ
            ing_Z = vs.zprefZ / thetaZ
            npp = u_P * phytoplankton_in  # Net production?????

            # Grazing is based on availability of food and eaters??

            # Grazing on P (phytoplankton?)
            g_P = gmax * ing_P * phytoplankton_in
            graz = g_P * zooplankton_in

            # Grazing on Z (zooplankton?)
            g_Z = gmax * ing_Z * zooplankton_in
            graz_Z = g_Z * zooplankton_in

            # Grazing on Detritus
            g_Det = gmax * ing_Det * detritus_in
            graz_Det = g_Det * zooplankton_in

            morp = vs.specific_mortality_phytoplankton * phytoplankton_in  # mortality of phytoplankton
            morpt = nupt * phytoplankton_in  # fast-recycling mortality of phytoplankton
            morz = vs.quadric_mortality_zooplankton * zooplankton_in ** 2  # mortality of zooplankton
            remineralization_detritus = nud * bct * detritus_in  # remineralization of detritus
            expo = wd[:, :, k] * detritus_in  # temporary detritus export

            # multiply by flags to ensure stability / not loosing more than there is
            # I think
            graz *= phytoflag * zooflag
            graz_Z *= zooflag
            graz_Det *= detrflag * zooflag
            morp *= phytoflag
            morpt *= phytoflag
            morz *= zooflag
            remineralization_detritus *= detrflag
            expo *= detrflag
            npp *= po4flag

            # -------------------------
            # Grazing: Digestion
            # -------------------------

            dig_P = vs.assimilation_efficiency * graz
            dig_Z = vs.assimilation_efficiency * graz_Z
            dig_Det = vs.assimilation_efficiency * graz_Det
            dig = dig_P + dig_Z + dig_Det

            # -------------------------
            # Grazing: Excretion
            # --------------------------

            excr_P = vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * graz
            excr_Z = vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * graz_Z
            excr_Det = vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * graz_Det
            excr = excr_P + excr_Z + excr_Det

            # -------------------------
            # Grazing: Sloppy feeding
            # -------------------------

            sf_P = (1 - vs.assimilation_efficiency) * graz
            sf_Z = (1 - vs.assimilation_efficiency) * graz_Z
            sf_Det = (1 - vs.assimilation_efficiency) * graz_Det
            sf = sf_P + sf_Z + sf_Det

            # -------------------------
            # Nutrients equation
            # -------------------------

            po4_in[:, :] += vs.dt_bio * vs.redfield_ratio_PN * (remineralization_detritus + excr - npp + morpt)

            # -------------------------
            # Phytoplankton equation
            # -------------------------

            phytoplankton_in[:, :] += vs.dt_bio * (npp - morp - graz - morpt)

            # ------------------------
            # Zooplankton equation
            # ------------------------

            zooplankton_in[:, :] += vs.dt_bio * (dig - morz - graz_Z - excr)

            # ------------------------
            # Detritus equation
            # ------------------------
            detritus_in[:, :] += vs.dt_bio * (morp + sf + morz - remineralization_detritus - graz_Det - expo + detritus_import)

            # update export from layer and flags
            detritus_export[:, :] += expo * vs.dt_bio  # NOTE the additional factor is only for easier debugging. It could be removed by removing the same factor in the import


            # NOTE: This would cause issues, when depleting resources too fast
            #       It would not preserve mass
            po4flag = po4flag * (po4_in > vs.trcmin)
            phytoflag = phytoflag * (phytoplankton_in > vs.trcmin)
            zooflag = zooflag * (zooplankton_in > vs.trcmin)
            detrflag = detrflag * (detritus_in > vs.trcmin)



            ## TODO: Should it be here?
            detritus_export = np.minimum(detritus_in, detritus_export)
            po4_in = np.maximum(po4_in, vs.trcmin)
            phytoplankton_in = np.maximum(phytoplankton_in, vs.trcmin)
            zooplankton_in = np.maximum(zooplankton_in, vs.trcmin)
            detritus_in = np.maximum(detritus_in, vs.trcmin)


        """
        End the npzd ecosystem dynamic model
        --------------------------------------
        """




        # TODO: Here should be source/sink terms and export of detritus updated

        # Source/sink terms
        # FIXME rdtts
        # detritus_out[:, :] *= .1  # rdtts????
        # phytoplankton_out[:, :] *= .1  # rdtts????
        # zooplankton_out[:, :] *= .1  # rdtts????
        # po4_out[:, :] *= .1  # rdtts????
        detritus_export[:, :] *= 1.0/nbio  # FIXME inefficient

        # ---------------------------------------------------
        # Calculate detritus at the bottom and remineralize
        # ---------------------------------------------------

        # TODO What is sgb_in?
        # It has something to do with topography. It must be 1 at the bottom for conservation
        # should it be here?
        # sgb_in = np.zeros_like(vs.detritus)
        # sgb_in[:, :, 3:] = 0
        # po4_in += sgb_in[:, :, k] * detritus_export * vs.redfield_ratio_PN
        # detritus_export[:, :] -= sgb_in[:, :, k] * detritus_export

        # ---------------------------------------------------
        # Set source/sink terms
        # ---------------------------------------------------

        # Calculate total export to get total import for next layer
        detritus_export *= vs.dzt[::-1][k]


        # NOTE Added to get values back into veros
        vs.po4[:, :, k] = po4_in[:, :]
        vs.phytoplankton[:, :, k] = phytoplankton_in[:, :]
        vs.zooplankton[:, :, k] = zooplankton_in[:, :]
        vs.detritus[:, :, k] = detritus_in[:, :]
