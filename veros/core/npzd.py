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


    # TODO the ones below should be set to settings just once..
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
    vs.saturation_constant_P = vs.saturation_constant_N * vs.redfield_ratio_PN

    for k in range(vs.phytoplankton.shape[2]):
        # print("layer", k)

        # incomming radiation at layer
        swr = swr * np.exp(- vs.light_attenuation_phytoplankton * phyto_integrated)
        phyto_integrated = np.maximum(vs.phytoplankton[:, :, k], vs.trcmin) * vs.dzt[::-1][k]
        phyto_integrated += np.maximum(vs.diazotroph[:, :, k], vs.trcmin) * vs.dzt[::-1][k]
        grid_light = swr * np.exp(ztt[k] * vs.rctheta)  # light at top of grid box

        # calculate detritus import pr time step from layer above
        detritus_import = detritus_export / vs.dzt[::-1][k] / vs.dt_bio

        # reset export
        detritus_export[:, :] = 0

        # TODO What is this?
        bct = vs.bbio ** (vs.cbio * vs.temp[:, :, k, vs.taum1])
        bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, k, vs.taum1], 20))  # TODO: Remove magic number
        nud = vs.nud0
        nudon = vs.nudon0
        nudop = vs.nudop0


        """
        --------------------------------------
        Call the npzd ecosystem dynamics model
        """

        po4_in = vs.po4[:, :, k]
        phytoplankton_in = vs.phytoplankton[:, :, k]
        zooplankton_in = vs.zooplankton[:, :, k]
        detritus_in = vs.detritus[:, :, k]
        no3_in = vs.no3[:, :, k]
        dop_in = vs.dop[:, :, k]
        don_in = vs.don[:, :, k]
        diazotroph_in = vs.diazotroph[:, :, k]

        # flags to prevent more outgoing flux than available capacity - stability?
        po4flag = po4_in > vs.trcmin
        phytoflag = phytoplankton_in > vs.trcmin
        zooflag = zooplankton_in > vs.trcmin
        detrflag = detritus_in > vs.trcmin
        no3flag = no3_in > vs.trcmin
        dopflag = dop_in > vs.trcmin
        donflag = don_in > vs.trcmin
        diazflag = diazotroph_in > vs.trcmin

        po4_in = np.maximum(po4_in, vs.trcmin)
        phytoplankton_in = np.maximum(phytoplankton_in, vs.trcmin)
        zooplankton_in = np.maximum(zooplankton_in, vs.trcmin)
        detritus_in = np.maximum(detritus_in, vs.trcmin)
        no3_in = np.maximum(no3_in, vs.trcmin)
        dop_in = np.maximum(dop_in, vs.trcmin)
        don_in = np.maximum(don_in, vs.trcmin)
        diazotroph_in = np.maximum(diazotroph_in, vs.trcmin)


        # ----------------------------
        # Fotosynthesis
        # After Evans & Parslow (1985)
        # ----------------------------
        light_attenuation = (vs.light_attenuation_water + vs.light_attenuation_phytoplankton * (phytoplankton_in + diazotroph_in)) * vs.dzt[::-1][k]
        f1 = np.exp(-light_attenuation)

        # TODO what are these things supposed to be?
        jmax = vs.abio_P * bct
        gd = jmax * vs.dt_mom # *(vs.dt_mom / 86400)
        avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

        # TODO remove magic number 2.6
        jmax_D = np.maximum(0, vs.abio_P * (bct - 2.6)) * vs.jdiar
        # TODO remove magic number 1e-14
        gd_D = np.maximum(1e-14, jmax_D) * vs.dt_mom#* (vs.dt_mom / 84600)
        avej_D = avg_J(vs, f1, gd_D, grid_light, light_attenuation)

        # -----------------------------
        # Grazing
        # -----------------------------

        # Maximum grazing rate is a function of temperature
        # bctz sets an upper limit on effects of temperature on grazing
        # phytoplankon growth rates bct are unlimited
        # FIXME: These quantities are not well defined or explained...
        gmax = vs.gbio * bctz
        nupt = vs.nupt0 * bct
        nupt_D = vs.nupt0_D * bct


        for _ in range(nbio):

            # TODO: What do these do? What do the variables mean?
            # TODO saturation / redfield is constant -> calculate only once
            # TODO separate into function (until Michaelis-Menten denominator
            limP_dop = vs.hdop * dop_in / (vs.saturation_constant_P + dop_in)
            limP_po4 = po4_in / (vs.saturation_constant_P + po4_in)

            sat_red = vs.saturation_constant_N / vs.redfield_ratio_PN

            dop_uptake_flag = limP_dop > limP_po4
            # nitrate limitation
            limP = po4_in / (sat_red + po4_in)
            u_P = np.minimum(avej, jmax * limP)
            u_P = np.minimum(u_P, jmax * no3_in / (sat_red + no3_in))
            u_D = np.minimum(avej_D, jmax_D * limP)

            dop_uptake_D_flag = dop_uptake_flag  # TODO should this be here?

            po4P = jmax * po4_in / (sat_red + po4_in)
            no3_P = jmax * no3_in / (sat_red + no3_in)
            po4_D = jmax_D * po4_in / (sat_red + po4_in)

            # Michaelis-Menten denominator
            thetaZ = vs.zprefP * phytoplankton_in + vs.zprefDet * detritus_in + vs.zprefZ * zooplankton_in + vs.zprefD * diazotroph_in + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN


            # ingestion of the different substances...?
            ing_P = vs.zprefP / thetaZ
            ing_Det = vs.zprefDet / thetaZ
            ing_Z = vs.zprefZ / thetaZ
            ing_D = vs.zprefD / thetaZ

            # Net primary production
            npp = u_P * phytoplankton_in
            npp_D = np.maximum(0, u_D * diazotroph_in)

            dop_uptake = npp * dop_uptake_flag

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

            # Grazing on D (diazotrophs)
            g_D = gmax * ing_D * diazotroph_in
            graz_D = g_D * zooplankton_in

            morp = vs.specific_mortality_phytoplankton * phytoplankton_in  # mortality of phytoplankton
            morpt = nupt * phytoplankton_in  # fast-recycling mortality of phytoplankton
            morp_D = vs.specific_mortality_diazotroph * diazotroph_in  # mortality of diazotrophs
            morpt_D = nupt_D * diazotroph_in  # fast-recycling of diazotrophs
            morz = vs.quadric_mortality_zooplankton * zooplankton_in ** 2  # mortality of zooplankton
            remineralization_detritus = nud * bct * detritus_in  # remineralization of detritus

            recycled_don = nudon * bct * don_in
            recycled_dop = nudop * bct * dop_in
            no3_uptake_D = (0.5 + 0.5 * np.tanh(no3_in - 5.0)) * npp_D  # nitrate uptake
            dop_uptake_D = npp_D * dop_uptake_D_flag


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

            # zooplankton change feeding based on availability
            npp *= no3flag * (dop_uptake_flag + (1 - dop_uptake_flag) * po4flag)

            # diazotrophs change feeding based on availability
            npp_D *= dop_uptake_D_flag * dopflag + (1 - dop_uptake_D_flag) * po4flag
            graz_D *= diazflag * zooflag
            morpt_D *= diazflag
            morp_D *= diazflag
            no3_uptake_D *= no3flag
            recycled_don *= donflag
            recycled_dop *= dopflag


            # -------------------------
            # Grazing: Digestion
            # -------------------------

            dig_P = vs.assimilation_efficiency * graz
            dig_Z = vs.assimilation_efficiency * graz_Z
            dig_Det = vs.assimilation_efficiency * graz_Det
            dig_D = vs.assimilation_efficiency * graz_D * (vs.redfield_ratio_PN / vs.diazotroph_NP)
            dig = dig_P + dig_Z + dig_Det + dig_D

            # -------------------------
            # Grazing: Excretion
            # --------------------------

            excr_P = vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * graz
            excr_Z = vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * graz_Z
            excr_Det = vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * graz_Det
            excr_D = vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * graz_D * (vs.redfield_ratio_PN / vs.diazotroph_NP)

            nr_excr_D = vs.assimilation_efficiency * graz_D * (1 - vs.redfield_ratio_PN / vs.diazotroph_NP) + (1 - vs.assimilation_efficiency) * graz_D * (1 - vs.redfield_ratio_PN / vs.diazotroph_NP)
            # FIXME is it just me, or is there too much math here?

            excr = excr_P + excr_Z + excr_Det + excr_D

            # -------------------------
            # Grazing: Sloppy feeding
            # -------------------------

            sf_P = (1 - vs.assimilation_efficiency) * graz
            sf_Z = (1 - vs.assimilation_efficiency) * graz_Z
            sf_Det = (1 - vs.assimilation_efficiency) * graz_Det
            sf_D = (1 - vs.assimilation_efficiency) * graz_D * (vs.redfield_ratio_PN / vs.diazotroph_NP)
            sf = sf_P + sf_Z + sf_Det + sf_D

            # -------------------------
            # Nutrients equation
            # -------------------------

            po4_in[:, :] += vs.dt_bio * (vs.redfield_ratio_PN * (remineralization_detritus + excr - npp + (1 - vs.dfrt) * morpt - (npp - dop_uptake)) + vs.diazotroph_NP * (morpt_D - (npp_D - dop_uptake_D)) + recycled_dop)

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

            detritus_in[:, :] += vs.dt_bio * ((1 - vs.dfr) * morp + sf + morz - remineralization_detritus - graz_Det + morp_D * (vs.redfield_ratio_PN / vs.diazotroph_NP)- expo + detritus_import)

            # ------------------------
            # Diazotroph equation
            # ------------------------

            diazotroph_in[:, :] += vs.dt_bio * (npp_D - morp_D - morpt_D - graz_D)

            # ------------------------
            # DON equation
            # ------------------------

            don_in[:, :] += vs.dt_bio * (vs.dfr * morp + vs.dfrt * morpt - recycled_don)

            # ------------------------
            # DOP equation
            # ------------------------

            dop_in[:, :] += vs.dt_bio * (vs.redfield_ratio_PN * (vs.dfr * morp + vs.dfrt * morpt - dop_uptake) - vs.diazotroph_NP * dop_uptake_D - recycled_dop)

            # ------------------------
            # Nitrate equation
            # ------------------------

            no3_in[:, :] += vs.dt_bio * (excr + remineralization_detritus + (1 - vs.dfrt) * morpt - npp + morpt_D - no3_uptake_D + recycled_don + nr_excr_D + morp_D * (1 - (vs.redfield_ratio_PN / vs.diazotroph_NP)))



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

        # Calculate total export to get total import for next layer
        detritus_export[:, :] *= 1.0/nbio
        detritus_export *= vs.dzt[::-1][k]

        # ---------------------------------------------------
        # Set source/sink terms
        # ---------------------------------------------------

        # NOTE Added to get values back into veros
        # TODO can't we just work directly on the values?
        vs.po4[:, :, k] = po4_in[:, :]
        vs.phytoplankton[:, :, k] = phytoplankton_in[:, :]
        vs.zooplankton[:, :, k] = zooplankton_in[:, :]
        vs.detritus[:, :, k] = detritus_in[:, :]
        vs.dop[:, :, k] = dop_in[:, :]
        vs.don[:, :, k] = don_in[:, :]
        vs.no3[:, :, k] = no3_in[:, :]
        vs.diazotroph[:, :, k] = diazotroph_in[:, :]


@veros_method
def avg_J(vs, f1, gd, grid_light, light_attenuation):
    """Average J"""
    u1 = np.maximum(grid_light / gd, 1e-6)  # TODO remove magic number 1e-6
    u2 = u1 * f1

    # NOTE: There is an approximation here: u1 < 20 WTF? Why 20?
    phi1 = np.log(u1 + np.sqrt(1 + u1**2)) - (np.sqrt(1 + u1) - 1) / u1
    phi2 = np.log(u2 + np.sqrt(1 + u2**2)) - (np.sqrt(1 + u2) - 1) / u2

    return gd * (phi1 - phi2) / light_attenuation
