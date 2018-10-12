"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
import numpy as np  # NOTE np is already defined somehow
from .. import veros_method
from . import advection, diffusion, thermodynamics, cyclic


@veros_method
def biogeochemistry(vs):
    """
    Integrate biochemistry: phytoplankton, zooplankton, detritus, po4
    """

    # Import and export of detritus in currently evaluated layer
    detritus_export = np.zeros((vs.detritus.shape[:2]))
    detritus_import = np.zeros_like(detritus_export)
    caco3_export = np.zeros((vs.caco3.shape[:2]))
    caco3_import = np.zeros_like(caco3_export)
    fe_export = np.zeros((vs.fe.shape[:2]))
    fe_import = np.zeros_like(fe_export)

    # Integrated phytplankton
    phyto_integrated = np.zeros_like(vs.phytoplankton[:, :, 0])

    # sinking speed of detritus
    # can't sink beyond bottom
    vs.wd = np.empty_like(vs.detritus)
    vs.wd[:, :] = (vs.wd0 + vs.mw * np.where(-vs.zw < vs.mwz, -vs.zw, vs.mwz)) / vs.dzt
    vs.wd *= vs.maskT

    # Sinking speed of calcite?
    vs.wc = np.empty_like(vs.caco3)
    vs.wc[:, :] = (vs.wc0 + vs.mw_c * np.where(-vs.zw < vs.mwz, -vs.zw, vs.mwz)) / vs.dzt
    vs.wc *= vs.maskT

    # if vs.enable_caco3:
    atmpres = 1.0

    nbio = int(vs.dt_mom // vs.dt_bio)

    vs.saturation_constant_P = vs.saturation_constant_N * vs.redfield_ratio_PN
    vs.redfield_ratio_OC = vs.redfield_ratio_ON / vs.redfield_ratio_PN
    sat_red = vs.saturation_constant_N / vs.redfield_ratio_PN

    # TODO what are these, and where do they belong?
    dic_npzd_sms = np.zeros(vs.nz)
    nfix = np.zeros_like(vs.diazotroph)


    # copy the incomming light forcing
    # we use this variable to decrease the available light
    # down the water column
    swr = vs.swr.copy()

    for k in reversed(range(vs.nz)):
    # for k in [vs.nz -1]:

        """

        Call some function co2calc_SWS


        """

        # print("layer", k)
        tracers = {name: value[:, :, k] for name, value in vs.npzd_tracers.items()}

        # flags to prevent more outgoing flux than available capacity - stability?
        flags = {tracer: True for tracer in tracers}

        # Set flags and tracers based on minimum concentration
        update_flags_and_tracers(vs, flags, tracers, refresh=True)

        # incomming radiation at layer
        swr = swr * np.exp(- vs.light_attenuation_phytoplankton * phyto_integrated
                           - vs.light_attenuation_caco3 * tracers["caco3"])

        phyto_integrated = np.maximum(tracers["phytoplankton"], vs.trcmin) * vs.dzt[k]
        phyto_integrated += np.maximum(tracers["diazotroph"], vs.trcmin) * vs.dzt[k]
        phyto_integrated += np.maximum(tracers["coccolitophore"], vs.trcmin) * vs.dzt[k]

        grid_light = swr * np.exp(vs.zw[k] * vs.rctheta)  # light at top of grid box




        """ Vars for iron
            Set them all to one to ignore iron
        """
        iron_limits = {"phytoplankton": (vs.kfemin, vs.kfemax),
                       "coccolitophore": (vs.kfemin_C, vs.kfemax_C),
                       "diazotroph": (vs.kfe_D, vs.kfe_D)}
        pmax = {"phytoplankton": vs.pmax,
                "coccolitophore": vs.pmax_C,
                "diazotroph": 1}  # TODO not 1, but some number

        gl = {"phytoplankton": 0, "coccolitophore": 0, "diazotroph": 0}
        deffe = {"phytoplankton": 0, "coccolitophore": 0, "diazotroph": 0}

        for plankton in gl:
            p1 = np.minimum(tracers[plankton], pmax[plankton])
            p2 = np.minimum(0, tracers[plankton] - pmax[plankton])
            kfevar = (iron_limits[plankton][0] * p1 + iron_limits[plankton][1] * p2) / (p1 + p2)
            deffe[plankton] = tracers["fe"] / (kfevar + tracers["fe"])
            thetamax = vs.thetamaxlo + (vs.thetamaxhi - vs.thetamaxlo) * deffe[plankton]
            alpha = vs.alphamin + (vs.alphamax - vs.alphamin) * deffe[plankton]
            gl[plankton] = grid_light * thetamax * alpha

        # p1 = np.minimum(tracers["phytoplankton"], vs.pmax)
        # p2 = np.maximum(0, tracers["phytoplankton"] - vs.pmax)
        # kfevar = (vs.kfemin * p1 + vs.kfemax * p2) / (p1 + p2)
        # deffe = tracers["fe"] / (kfevar + tracers["fe"])
        # thetamax = vs.thetamaxlo + (vs.thetamaxhi - vs.thetamaxlo) * deffe
        # alpha_O = vs.alphamin + (vs.alphamax - vs.alphamin) * deffe
        # gl_O = grid_light * thetamax * alpha_O

        # # if vs.enable_caco3:
        # p1_C = np.minimum(tracers["coccolitophore"], vs.pmax_C)
        # p2_C = np.maximum(0, tracers["coccolitophore"] - vs.pmax_C)
        # kfevar_C = (vs.kfemin_C * p1_C + vs.kfemax * p2_C) / (p1_C + p2_C)
        # deffe_C = tracers["fe"] / (kfevar_C + tracers["fe"])
        # thetamax_C = vs.thetamaxlo + (vs.thetamaxhi - vs.thetamaxlo) * deffe_C
        # alpha_OC = vs.alphamin + (vs.alphamax - vs.alphamin) * deffe_C
        # gl_C = grid_light * thetamax_C * alpha_OC

        # # if vs.enable_N:
        # deffe_D = tracers["fe"] / (vs.kfe_D + tracers["fe"])
        # thetamax_D = vs.thetamaxlo + (vs.thetamaxhi - vs.thetamaxlo) * deffe_D
        # alpha_D = vs.alphamin + (vs.alphamax - vs.alphamin) * deffe_D
        # gl_D = grid_light * thetamax_D * alpha_D


        """ End vars for iron
        """









        # calculate detritus import pr time step from layer above, reset export
        detritus_import = detritus_export / vs.dzt[k] / vs.dt_bio
        detritus_export[:, :] = 0
        # calculate caco3 import pr time step from layer above, reset export
        caco3_import = caco3_export / vs.dzt[k] / vs.dt_bio
        caco3_export[:, :] = 0
        # calculate iron import pr time step from layer above, reset export
        fe_import = fe_export / vs.dzt[k] / vs.dt_bio
        fe_export[:, :] = 0

        dissk1 = vs.dissk0 * (1 - vs.Omega_c)

        # TODO What is this?
        bct = vs.bbio ** (vs.cbio * vs.temp[:, :, k, vs.taum1])
        bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, k, vs.taum1], 20))  # TODO: Remove magic number

        # These are left here for later expansion, when they may change
        # based on available stuff...
        nud = vs.nud0
        nudon = vs.nudon0
        nudop = vs.nudop0
        nupt = vs.nupt0
        nupt_D = vs.nupt0_D
        nuct = vs.nuct0

        # decrease remineralisation rate in oxygen minimum zone
        # TODO remove magic numbers
        bctz *= (0.5 * (np.tanh(tracers["o2"] - 8.0) + 1))
        nud *= (0.65 + 0.35 * np.tanh(tracers["o2"] - 3.0))


        # Maximum grazing rate is a function of temperature
        # bctz sets an upper limit on effects of temperature on grazing
        gmax = vs.gbio * bctz

        # recyclers include fast recycling of plankton, remineralization of detritus and recycling of dissolved organic matter
        # This may be incorrect as some of the fast-recycled phytoplankton becomes DOP and DON, which is also included in this list
        recycling_rates = {"phytoplankton": nupt * bct,
                           "diazotroph": nupt_D * bct,
                           "detritus": nud * bct,
                           "DON": nudon * bct,
                           "DOP": nudop * bct,
                           "coccolitophore": nuct * bct}



        # ----------------------------
        # Fotosynthesis
        # After Evans & Parslow (1985)
        # ----------------------------
        light_attenuation = (vs.light_attenuation_water + vs.light_attenuation_phytoplankton * (tracers["phytoplankton"] + tracers["diazotroph"] + tracers["coccolitophore"]) + vs.light_attenuation_caco3 * tracers["caco3"]) * vs.dzt[k]
        f1 = np.exp(-light_attenuation)

        # NOTE there are different minimum values in avej for phytoplankton and diazotroph - why?
        # TODO remove magic number 1e-14
        # TODO remove magic number 2.6
        # TODO dayfrac? dt_mom, dt_bio? noget andet?
        # jmax = {"phytoplankton": vs.abio_P * bct
        #         "diazotroph": np.maximum(0, vs.abio_P * (bct - 2.6)) * vs.jdiar,
        #         "coccolitophore": vs.abio_C * bct}
        # gd = {"phytoplankton": jmax["phytoplankton"] * vs.dt_mom,
        #       "diazotroph": np.maximum(1.e-14, jmax["diazotroph"]) * vs.dt_mom,
        #       "coccolitophore": jmax["coccolitophore"] * vs.dt_mom}
        # avej = {"phytoplankton": avg_J(vs, f1, gd["phytoplankton"], grid_light, light_attenuation),
        #         "diazotroph": avg_J(vs, f1, gd["diazotroph"], grid_light, light_attenuation),
        #         "coccolitophore": avg_J(vs, f1, gd["coccolitophore"], grid_light, light_attenuation)}
        jmax = {"phytoplankton": vs.abio_P * bct * deffe["phytoplankton"],
                "diazotroph": np.maximum(0, vs.abio_P * (bct - 2.6) * deffe["phytoplankton"]) * vs.jdiar,
                "coccolitophore": vs.abio_C * bct * deffe["coccolitophore"]}
        gd = {"phytoplankton": jmax["phytoplankton"] * vs.dt_mom,
              "diazotroph": np.maximum(1.e-14, jmax["diazotroph"]) * vs.dt_mom,
              "coccolitophore": jmax["coccolitophore"] * vs.dt_mom}
        avej = {"phytoplankton": avg_J(vs, f1, gd["phytoplankton"], gl["phytoplankton"], light_attenuation),
                "diazotroph": avg_J(vs, f1, gd["diazotroph"], gl["diazotroph"], light_attenuation),
                "coccolitophore": avg_J(vs, f1, gd["coccolitophore"], gl["coccolitophore"], light_attenuation)}


        for _ in range(nbio):

            """

            Fe bullshit
            """
            p1 = np.minimum(tracers["phytoplankton"], vs.pmax)
            p2 = np.maximum(0, tracers["phytoplankton"] - vs.pmax)
            kfevar = (vs.kfemin * p1 + vs.kfemax * p2) / (p1 + p2)
            deffe = tracers["fe"] / (kfevar + tracers["fe"])
            jmax["phytoplankton"] = vs.abio_P * bct * deffe
            p1 = np.minimum(tracers["coccolitophore"], vs.pmax_C)
            p2 = np.maximum(0, tracers["coccolitophore"] - vs.pmax_C)
            kfevar = (vs.kfemin * p1 + vs.kfemax * p2) / (p1 + p2)
            deffe = tracers["fe"] / (kfevar + tracers["fe"])
            jmax["coccolitophore"] = vs.abio_P * bct * deffe
            deffe = tracers["fe"] / (vs.kfe_D + tracers["fe"])
            jmax["diazotroph"] = np.maximum(0, vs.abio_P * (bct - 2.6) * deffe) * vs.jdiar



            """
            growth rate of phytoplankton
            consume DOP when it is more efficient
            """
            limP_dop = vs.hdop * tracers["DOP"] / (sat_red + tracers["DOP"])
            limP_po4 = tracers["po4"] / (sat_red + tracers["po4"])
            dopupt_flag = limP_dop > limP_po4
            limP = limP_dop * dopupt_flag + limP_po4 * np.logical_not(dopupt_flag)

            # These values are defined but never used...
            # po4P = jmax["phytoplankton"] * tracers["po4"] / (sat_red + tracers["po4"])
            # no3P = jmax["phytoplankton"] * tracers["no3"] / (vs.saturation_constant_N + tracers["no3"])
            # po4_D = jmax["diazotroph"] * tracers["po4"] / (sat_red + tracers["po4"])

            limP_dop_C = vs.hdop * tracers["DOP"] / (vs.saturation_constant_NC * vs.redfield_ratio_PN + tracers["DOP"])
            limP_po4_C = tracers["po4"] / (vs.saturation_constant_NC * vs.redfield_ratio_PN + tracers["po4"])
            dopupt_C_flag = limP_dop_C > limP_po4_C
            limP_C = limP_dop_C * dopupt_C_flag + limP_po4_C * np.logical_not(dopupt_C_flag)

            net_primary_production = {"phytoplankton": 0, "diazotroph": 0, "coccolitophore": 0}

            for producer in net_primary_production:
                if producer == "cooclitophore":
                    u = np.minimum(avej[producer], jmax[producer] * limP_C)
                else:
                    u = np.minimum(avej[producer], jmax[producer] * limP)

                if producer == "phytoplankton":# and "no3" in tracers:
                    # Phytoplankton growth is nitrate limited
                    u = np.minimum(u, jmax[producer] * tracers["no3"] /
                                   (vs.saturation_constant_N + tracers["no3"]))
                elif producer == "coccolitophore":
                    u = np.minimum(u, jmax[producer] * tracers["no3"] /
                                   (vs.saturation_constant_NC + tracers["no3"]))

                # only diazotrophs had 0 minimum
                # but does it make sense to have negative primary production?
                net_primary_production[producer] = np.maximum(0, u * tracers[producer])



            """
            [5] Denitrification occurs under suboxic conditions (O2 <
            5 mmol/kg) in the water column and in the seafloor sedi-
            −ments. Here, microbes use NO3 instead of O2 as the electron
            acceptor during respiration and convert it to gaseous forms of
            N (N2O and N2), which can then escape to the atmosphere
            [Codispoti and Richards, 1976]. The volume and distribution
            of suboxic water is affected by the temperature‐dependent
            sCagmpc
            Somes, Schmittner 
            """

            dop_uptake = net_primary_production["phytoplankton"] * dopupt_flag
            no3_uptake_D = (.5 + .5 * np.tanh(tracers["no3"] - 5.)) * net_primary_production["diazotroph"]  # nitrate uptake TODO: remove magic number 5
            dop_uptake_D = net_primary_production["diazotroph"] * dopupt_flag
            dop_uptake_C = net_primary_production["coccolitophore"] * dopupt_C_flag

            # Michaelis-Menten denominator
            thetaZ = sum([pref_score * tracers[preference] for preference, pref_score in vs.zooplankton_preferences.items()]) + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

            # Ingestion by zooplankton based on preference
            ingestion = {preference: pref_score / thetaZ for preference, pref_score in vs.zooplankton_preferences.items()}

            # Grazing is based on availability of food and eaters??
            grazing = {preference: flags[preference] * flags["zooplankton"] * gmax *
                       ingestion[preference] * tracers[preference] * tracers["zooplankton"]
                       for preference in vs.zooplankton_preferences}

            mortality = {plankton: flags[plankton] * rate * tracers[plankton]
                         for plankton, rate in vs.mortality_rates.items()}
            mortality["zooplankton"] *= tracers["zooplankton"]  # quadric rate

            recycled = {recycler: flags[recycler] * rate * tracers[recycler]
                        for recycler, rate in recycling_rates.items()}

            net_primary_production["phytoplankton"] *= flags["no3"] * (flags["DOP"] * dopupt_flag + flags["po4"] * np.logical_not(dopupt_flag))
            net_primary_production["diazotroph"] *= flags["no3"] * (flags["DOP"] * dopupt_flag + flags["po4"] * np.logical_not(dopupt_flag))
            net_primary_production["coccolitophore"] *= flags["no3"] * (flags["DOP"] * dopupt_C_flag + flags["po4"] * np.logical_not(dopupt_C_flag))


            dissl = tracers["caco3"] * dissk1

            expo = vs.wd[:, :, k] * tracers["detritus"]  # temporary detritus export
            expo *= flags["detritus"]

            expo_caco3 = vs.wc[:, :, k] * tracers["caco3"]  # temporary caco3 export
            expo_caco3 *= flags["caco3"]


            """ More iron bullshit ... this guy ..
            """
            par = 0.43 # fraction of photosynthetically active radiation
            irrtop = grid_light / 2 / par
            kirr = - light_attenuation
            # aveirr = -1 / vs.dzt / kirr * (irrtop - irrtop * np.exp(kirr * vs.dzt))
            remife = nud * bct * tracers["particulate_fe"] * flags["particulate_fe"]
            # Scavenging of dissolved iron is based on Honeymoon et al. (1988) 
            # and Parekh et al. (2004). 
            flags["o2"] = tracers["o2"] > vs.o2min
            fepa = (1.0 - vs.kfeleq * (vs.lig - tracers["fe"])) * flags["o2"]
            feprime = ((-fepa + (fepa * fepa + 4.0 * vs.kfeleq * tracers["fe"]) ** 0.5) / (2.0 * vs.kfeleq)) * flags["o2"]
            feorgads = (vs.kfeorg * (((tracers["detritus"] * flags["detritus"]) * vs.mc * vs.redfield_ratio_PN * vs.redfield_ratio_CP)**0.58) * feprime) * flags["o2"] * flags["particulate_fe"]
            fecol = vs.kfecol * feprime * flags["o2"] * flags["fe"]

            expo_fe = vs.wd[:, :, k] * tracers["particulate_fe"] * flags["particulate_fe"]

        
            # expo_fe = vs.wf[:, :, k] * tracers["fe"]  # temporary fe export
            # expo_fe *= flags["fe"]

            """
            Grazing is split into several parts:
            Digestion: How much of the eaten material is put into growing the population
            Excretion: Eaten metrial is disposed as nutrients. This amount is not available for growing
            Sloppy feeding: Material is not consumed by zooplankton and is lost as detritus
            """

            digestion = {preference: vs.assimilation_efficiency * amount_grazed for preference, amount_grazed in grazing.items()}
            digestion["diazotroph"] *= 1. / (vs.redfield_ratio_PN * vs.diazotroph_NP)
            digestion_total = sum(digestion.values())

            excretion = {preference: vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}
            excretion["diazotroph"] *= (vs.redfield_ratio_PN * vs.diazotroph_NP)
            excretion_total = sum(excretion.values())

            nr_excr_D = vs.assimilation_efficiency * grazing["diazotroph"] * (1 - 1./(vs.redfield_ratio_PN * vs.diazotroph_NP)) + (1 - vs.assimilation_efficiency) * grazing["diazotroph"] * (1 - 1./(vs.redfield_ratio_PN * vs.diazotroph_NP))
            # FIXME is it just me, or is there too much math here?

            sloppy_feeding = {preference: (1 - vs.assimilation_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}
            sloppy_feeding["diazotroph"] *= 1. / (vs.redfield_ratio_PN * vs.diazotroph_NP)
            sloppy_feeding_total = sum(sloppy_feeding.values())


            # net formation of attached (living tests
            # total primary production by coccolitophores + total growth of zooplankton
            calatt = (net_primary_production["coccolitophore"] - mortality["coccolitophore"] - recycled["coccolitophore"] - grazing["coccolitophore"] + digestion_total - mortality["zooplankton"] - grazing["zooplankton"] - excretion_total) * vs.capr * vs.redfield_ratio_CP * vs.redfield_ratio_PN * 1e3

            calpro = (sloppy_feeding["coccolitophore"] + sloppy_feeding["zooplankton"] + mortality["coccolitophore"] + mortality["zooplankton"]) * vs.capr * vs.redfield_ratio_CP * vs.redfield_ratio_PN * 1e3

            """
            Model dynamics:
            """

            tracers["po4"][:, :] += vs.dt_bio * (vs.redfield_ratio_PN * (recycled["detritus"] + excretion_total - net_primary_production["phytoplankton"] + (1 - vs.dfrt) * recycled["phytoplankton"] - (net_primary_production["phytoplankton"] - dop_uptake)) + 1. / vs.diazotroph_NP * (recycled["diazotroph"] - (net_primary_production["diazotroph"] - dop_uptake_D)) + recycled["DOP"] + (1 - vs.dfrt) * mortality["coccolitophore"] - (net_primary_production["coccolitophore"] + dop_uptake_C))

            tracers["phytoplankton"][:, :] += vs.dt_bio * (net_primary_production["phytoplankton"] - mortality["phytoplankton"] - grazing["phytoplankton"] - recycled["phytoplankton"])

            tracers["zooplankton"][:, :] += vs.dt_bio * (digestion_total - mortality["zooplankton"] - grazing["zooplankton"] - excretion_total)

            tracers["detritus"][:, :] += vs.dt_bio * ((1 - vs.dfr) * mortality["phytoplankton"] + sloppy_feeding_total + mortality["zooplankton"] - recycled["detritus"] - grazing["detritus"] + mortality["diazotroph"] / (vs.redfield_ratio_PN * vs.diazotroph_NP) - expo + detritus_import + (1 - vs.dfr) * mortality["coccolitophore"])  # TODO simplify mortality to use sum

            tracers["diazotroph"][:, :] += vs.dt_bio * (net_primary_production["diazotroph"] - mortality["diazotroph"] - recycled["diazotroph"] - grazing["diazotroph"])

            tracers["DON"][:, :] += vs.dt_bio * (vs.dfr * mortality["phytoplankton"] + vs.dfrt * recycled["phytoplankton"] + vs.dfr * mortality["coccolitophore"] + vs.dfrt * recycled["coccolitophore"]- recycled["DON"])

            tracers["DOP"][:, :] += vs.dt_bio * (vs.redfield_ratio_PN * (vs.dfr * mortality["phytoplankton"] + vs.dfrt * recycled["phytoplankton"] - dop_uptake) - 1. / vs.diazotroph_NP * dop_uptake_D - recycled["DOP"] + vs.dfr * mortality["coccolitophore"] + vs.dfrt * recycled["coccolitophore"] - dop_uptake_C)

            tracers["no3"][:, :] += vs.dt_bio * (excretion_total + recycled["detritus"] + (1 - vs.dfrt) * recycled["phytoplankton"] - net_primary_production["phytoplankton"] + recycled["diazotroph"] - no3_uptake_D + recycled["DON"] + nr_excr_D + mortality["diazotroph"] * (1 - 1./(vs.redfield_ratio_PN * vs.diazotroph_NP)) + (1 - vs.dfr) * mortality["coccolitophore"] - net_primary_production["coccolitophore"])

            tracers["coccolitophore"][:, :] += vs.dt_bio * (net_primary_production["coccolitophore"] - mortality["coccolitophore"] - grazing["coccolitophore"] - recycled["coccolitophore"])

            tracers["caco3"][:, :] += vs.dt_bio * (calpro - dissl - expo_caco3 + caco3_import)

            tracers["fe"][:, :] += vs.dt_bio * (vs.rfeton * (excretion_total + (1 - vs.dfrt) * recycled["phytoplankton"] - net_primary_production["phytoplankton"] + recycled["diazotroph"] - net_primary_production["diazotroph"] + recycled["DON"] + nr_excr_D + mortality["diazotroph"] * (1 - 1. / (vs.redfield_ratio_PN / vs.diazotroph_NP))) - feorgads + remife - fecol + vs.rfeton * ((1 - vs.dfrt) * recycled["coccolitophore"] - net_primary_production["coccolitophore"]))

            tracers["particulate_fe"][:, :] += vs.dt_bio * (vs.rfeton * (sloppy_feeding_total + (1 - vs.dfr) * mortality["phytoplankton"] + mortality["diazotroph"] * (vs.redfield_ratio_PN / vs.diazotroph_NP) + mortality["zooplankton"] - grazing["detritus"]) + feorgads + fecol - remife - expo_fe + fe_import + vs.rfeton * (1 - vs.dfr) * mortality["coccolitophore"])


            # update export from layer and flags
            detritus_export[:, :] += expo * vs.dt_bio  # NOTE the additional factor is only for easier debugging. It could be removed by removing the same factor in the import
            caco3_export[:, :] += expo_caco3 * vs.dt_bio
            fe_export[:, :] += expo_fe * vs.dt_bio

            nfix[:, :, k] += net_primary_production["diazotroph"] - no3_uptake_D

            ## TODO: Should it be here?
            # detritus_export = np.minimum(tracers["detritus"], detritus_export)
            update_flags_and_tracers(vs, flags, tracers)


        # !-----------------------------------------------------------------------
        # !        benthic denitrification model of Bohlen et al., 2012, GBC (eq. 10)
        # !        NO3 is removed out of bottom water nitrate.
        # !        See Somes et al., 2012, BGS for additional details/results
        # !-----------------------------------------------------------------------
        # TODO remove magic numbers
        limit_no3 = .5 * np.tanh(tracers["no3"] * 10 - 5.0)

        # TODO this should be simplieable as flags has just been updated
        sg_bdeni = (0.06 + 0.19 * 0.99**(np.maximum(tracers["o2"], vs.trcmin)
                                         - np.maximum(tracers["no3"], vs.trcmin)))

        # TODO sgb_in is already accounted for in sinking speed?
        # TODO what do these things mean?
        # TODO remove min, max blubber
        sg_bdeni = np.minimum(sg_bdeni, detritus_export)
        sg_bdeni = np.maximum(sg_bdeni, 0)
        sg_bdeni = sg_bdeni * (.5 + limit_no3) * flags["no3"] # * din15flag
        # bdeni[k] = sg_bdeni  # wat?
        tracers["no3"] += detritus_export - sg_bdeni
        tracers["po4"] += detritus_export * vs.redfield_ratio_PN

        fesed = vs.fetopsed * bct * vs.redfield_ratio_PN * detritus_export
        # bfe = vs.fetopsed
        tracers["fe"] += fesed
        flags["o2"] = tracers["o2"] - vs.o2min
        # tracers["fe"] += fe_export * sgb_in[k] * flags["o2"]
        tracers["fe"] += fe_export * vs.maskT[:, :, k] * flags["o2"]
        # bfe += fe_export * vs.maskT[k] * flags["o2"]
        # fe_export -= sgb_in[k] * fe_export * flags["o2"]
        fe_export -= vs.maskT[:, :, k] * fe_export * flags["o2"]

        # Calculate total export to get total import for next layer
        detritus_export[:, :] *= 1.0/nbio
        detritus_export *= vs.dzt[k]
        caco3_export[:, :] *= 1.0 / nbio * vs.dzt[k]
        fe_export[:, :] *= 1.0 /nbio * vs.dzt[k]

        # Update model values
        for name, value in tracers.items():
            vs.npzd_tracers[name][:, :, k] = value

        # apparently after updating
        # dic_npzd_sms[k] = tracers["po4"] * vs.redfield_ratio_CP

    # TODO is there any reason, this couldn't be in the other loop?
    for k in range(vs.nz):
        # limit oxygen consumption below concentration of 5umol/kg
        # as reccomended in OCMIP
        fo2 = .5 * np.tanh(vs.npzd_tracers["o2"][:, :, k] - 2.5)
        # sink of oxygen
        # O2 is needed to generate the equivalent of NO3 from N2 during N2 fixation
        # 0.5 H2O + 0.5 N2+1.25O2 -> HNO3
        # note that so2 is -dO2/dt
        # TODO remove magic number
        so2 = dic_npzd_sms[k] * vs.redfield_ratio_OC + nfix[:, :, k] * (1.25e-3 / nbio)
        flags["no3"] = vs.npzd_tracers["no3"] > vs.trcmin  # TODO why would this check be necessary?
        limit_no3 = .5 * np.tanh(vs.npzd_tracers["no3"][:, :, k] - 2.5)
        # limit_ntp = 0.5 * np.tanh(vs.npzd_tracers["no3"] / (vs.refield_ratio_NP * vs.npzd_tracers["po4"]) * 100 - 60.)
        # 800 = (elec/mol O2)/(elec/mol NO3)*(mmol/mol)
        wcdeni = 800 * flags["no3"][:, :, k] * so2 * (.5 - fo2) * (.5 * limit_no3)
        wcdeni = np.maximum(wcdeni, 0)
        vs.npzd_tracers["no3"][:, :, k] -= wcdeni


@veros_method
def avg_J(vs, f1, gd, grid_light, light_attenuation):
    """Average J"""
    u1 = np.maximum(grid_light / gd, 1e-6)  # TODO remove magic number 1e-6
    u2 = u1 * f1

    # NOTE: There is an approximation here: u1 < 20 WTF? Why 20?
    phi1 = np.log(u1 + np.sqrt(1 + u1**2)) - (np.sqrt(1 + u1) - 1) / u1
    phi2 = np.log(u2 + np.sqrt(1 + u2**2)) - (np.sqrt(1 + u2) - 1) / u2

    return gd * (phi1 - phi2) / light_attenuation


@veros_method
def update_flags_and_tracers(vs, flags, tracers, refresh=False):
    """Set flags"""

    for key in tracers:
        keep = flags[key] if not refresh else True  # if the flag was already false, keep it false
        flags[key] = (tracers[key] > vs.trcmin) * keep

    for key in tracers:
        tracers[key] = np.maximum(tracers[key], vs.trcmin)


@veros_method
def npzd(vs):

    tracer_result = {key: 0 for key in vs.npzd_tracers}

    for tracer, val in vs.npzd_tracers.items():
        dNPZD_advect = np.zeros_like(val)
        dNPZD_diff = np.zeros_like(val)

        # NOTE Why is this in thermodynamics?
        thermodynamics.advect_tracer(vs, val, dNPZD_advect)
        diffusion.biharmonic(vs, val, 0.5, dNPZD_diff)  # TODO correct parameter
        tracer_result[tracer] = vs.dt_mom * (dNPZD_advect + dNPZD_diff)

    biogeochemistry(vs)

    for tracer in vs.npzd_tracers:
        vs.npzd_tracers[tracer][...] += tracer_result[tracer]

    if vs.enable_cyclic_x:
        for tracer in vs.npzd_tracers.values():
            cyclic.setcyclic_x(tracer)
