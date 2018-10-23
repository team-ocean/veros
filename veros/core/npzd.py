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

    vs.mortality_rates["phytoplankton"] = vs.specific_mortality_phytoplankton
    vs.mortality_rates["zooplankton"] = vs.quadric_mortality_zooplankton
    vs.zooplankton_preferences["phytoplankton"] = vs.zprefP
    vs.zooplankton_preferences["zooplankton"] = vs.zprefZ
    vs.zooplankton_preferences["detritus"] = vs.zprefDet
    vs.plankton_types = ["phytoplankton"]

    from collections import defaultdict
    vs.sinking_speeds = defaultdict(None)

    if vs.enable_nitrogen:
        vs.zooplankton_preferences["diazotroph"] = vs.zprefD
        vs.mortality_rates["diazotroph"] = vs.specific_mortality_diazotroph
        vs.plankton_types = ["phytoplankton", "diazotroph"]

    if vs.enable_calcifiers:
        vs.zooplankton_preferences["coccolitophore"] = vs.zprefC
        vs.zooplankton_preferences["ballast"] = vs.zprefDet
        vs.mortality_rates["coccolitophore"] = vs.specific_mortality_coccolitophore
        vs.plankton_types = ["phytoplankton", "diazotroph", "coccolitophore"]

    saturation_constants = {"phytoplankton": vs.saturation_constant_N, "diazotroph": vs.saturation_constant_N, "coccolitophore": vs.saturation_constant_NC}

    growth_parameters = {"phytoplankton": vs.abio_P, "diazotroph": vs.abio_P * vs.jdiar, "coccolitophore": vs.abio_C}
    min_bct = {"phytoplankton": 0, "diazotroph": 2.6, "coccolitophore": 0}

    iron_limits = {"phytoplankton": (vs.kfemin, vs.kfemax), "coccolitophore": (vs.kfemin_C, vs.kfemax_C), "diazotroph": (vs.kfe_D, vs.kfe_D)}
    pmax = {"phytoplankton": vs.pmax, "coccolitophore": vs.pmax_C, "diazotroph": np.inf}


    if vs.enable_nitrogen:
        doms = ("DON", "DOP")
    elif vs.enable_calcifiers:
        doms = ("DON", "DOP", "DIC")
    else:
        doms = []


    # TODO correct for diazotroph
    uptake_factors = {"phytoplankton": vs.dfr, "diazotroph": vs.bapr, "coccolitophore": vs.dfr, "zooplankton": vs.dfr}
    nutrient_factors = {"po4": vs.redfield_ratio_PN, "no3": 1, "fe": vs.rfeton}

    # for storing denitrification for multiple loops....
    bdeni = np.zeros_like(vs.detritus)


    if vs.enable_calcifiers:
        dissl = np.zeros_like(vs.detritus)
        calatt = np.zeros_like(vs.detritus)
        calpro = np.zeros_like(vs.detritus)

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

    vs.sinking_speeds["detritus"] = vs.wd

    if vs.enable_calcifiers:
        vs.sinking_speeds["caco3"] = vs.wc
        vs.sinking_speeds["ballast"] = vs.wd

    if vs.enable_iron:
        vs.sinking_speeds["particulate_fe"] = vs.wc

    impo = {sinker: np.zeros((vs.detritus.shape[:2])) for sinker in vs.sinking_speeds}
    export = {sinker: np.zeros((vs.detritus.shape[:2])) for sinker in vs.sinking_speeds}

    nbio = int(vs.dt_mom // vs.dt_bio)

    vs.saturation_constant_P = vs.saturation_constant_N * vs.redfield_ratio_PN
    vs.redfield_ratio_OC = vs.redfield_ratio_ON / vs.redfield_ratio_PN
    sat_red = vs.saturation_constant_N / vs.redfield_ratio_PN
    vs.redfield_ratio_CN = vs.redfield_ratio_CP * vs.redfield_ratio_PN

    # TODO what are these, and where do they belong?
    # dic_npzd_sms = np.zeros(vs.nz)

    if vs.enable_nitrogen:
        nfix = np.zeros_like(vs.diazotroph)


    # copy the incomming light forcing
    # we use this variable to decrease the available light
    # down the water column
    swr = vs.swr.copy()

    for k in reversed(range(vs.nz)):

        tracers = {name: value[:, :, k] for name, value in vs.npzd_tracers.items()}

        # flags to prevent more outgoing flux than available capacity - stability?
        flags = {tracer: True for tracer in tracers}

        # Set flags and tracers based on minimum concentration
        update_flags_and_tracers(vs, flags, tracers, refresh=True)

        # incomming radiation at layer
        swr = swr * np.exp(- vs.light_attenuation_phytoplankton * phyto_integrated)

        if vs.enable_calcifiers:
            swr[:, :] *= np.exp(-vs.light_attenuation_caco3 * tracers["caco3"])

        phyto_integrated = sum([np.maximum(tracers[plankton], vs.trcmin) for plankton in vs.plankton_types]) * vs.dzt[k]

        grid_light = swr * np.exp(vs.zw[k] * vs.rctheta)  # light at top of grid box



        # calculate detritus import pr time step from layer above, reset export
        for sinker in vs.sinking_speeds:
            impo[sinker] = export[sinker] / (vs.dzt[k] * vs.dt_bio)
            export[sinker][:, :] = 0

        if vs.enable_calcifiers:
            # TODO What is va.Omega_c???
            dissk1 = vs.dissk0 * (1 - vs.Omega_c)

        # TODO What is this?
        bct = vs.bbio ** (vs.cbio * vs.temp[:, :, k, vs.taum1])
        bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, k, vs.taum1], 20))  # TODO: Remove magic number

        # decrease remineralisation rate in oxygen minimum zone
        # TODO remove magic numbers
        bctz *= (0.5 * (np.tanh(tracers["o2"] - 8.0) + 1))

        # Maximum grazing rate is a function of temperature
        # bctz sets an upper limit on effects of temperature on grazing
        gmax = vs.gbio * bctz

        # recyclers include fast recycling of plankton, remineralization of detritus and recycling of dissolved organic matter
        # This may be incorrect as some of the fast-recycled phytoplankton becomes DOP and DON, which is also included in this list
        vs.recycling_rates["phytoplankton"] = (0.65 + 0.35 * np.tanh(tracers["o2"] - 8.0) * vs.nupt0 * bct)

        # TODO is this correct
        vs.recycling_rates["detritus"] = vs.nupt0_D * bct

        if vs.enable_nitrogen:
            vs.recycling_rates["diazotroph"] = vs.nupt0_D * bct
            vs.recycling_rates["DON"] = vs.nudon0 * bct
            vs.recycling_rates["DOP"] = vs.nudop0 * bct

        if vs.enable_calcifiers:
            vs.recycling_rates["coccolitophore"] = vs.nuct0 * bct
        # TODO shouldn't there also be one for iron?

        if vs.enable_iron:
            vs.recycling_rates["particulate_fe"] = vs.recycling_rates["detritus"]


        # ----------------------------
        # Fotosynthesis
        # After Evans & Parslow (1985)
        # ----------------------------
        light_attenuation = vs.dzt[k] * (vs.light_attenuation_water + vs.light_attenuation_phytoplankton * sum([tracers[plankton] for plankton in vs.plankton_types]))
        # TODO there is a large overlap with this and swr

        if vs.enable_calcifiers:
            light_attenuation[:, :] += vs.light_attenuation_caco3 * tracers["caco3"]

        f1 = np.exp(-light_attenuation)

        jmax = {plankton: 0 for plankton in vs.plankton_types}
        gd = {plankton: 0 for plankton in vs.plankton_types}
        avej = {plankton: 0 for plankton in vs.plankton_types}
        gl = {plankton: 0 for plankton in vs.plankton_types}

        for plankton in gl:
            gl[plankton] = grid_light

            if vs.enable_iron:
                limFe = iron_limitation(vs, *iron_limits[plankton], tracers["fe"], tracers[plankton], pmax[plankton])
                thetamax = vs.thetamaxlo + (vs.thetamaxhi - vs.thetamaxlo) * limFe
                alpha = vs.alphamin + (vs.alphamax - vs.alphamin) * limFe
                gl[plankton] *= thetamax * alpha


        for plankton in vs.plankton_types:
            jmax[plankton] = np.maximum(0, growth_parameters[plankton] * (bct - min_bct[plankton]))
            # TODO Why 1e-14, it doesn't work without
            gd[plankton] = np.maximum(1e-14, jmax[plankton] * vs.dt_mom)

            if vs.enable_iron:
                limFe = iron_limitation(vs, *iron_limits[plankton], tracers["fe"], tracers[plankton], pmax[plankton])
                avej[plankton] = avg_J(vs, f1, gd[plankton] * limFe, gl[plankton], light_attenuation)
            else:
                avej[plankton] = avg_J(vs, f1, gd[plankton], gl[plankton], light_attenuation)


        for _ in range(nbio):

            net_primary_production = {plankton: 0 for plankton in vs.plankton_types}
            dop_uptake = {plankton: 0 for plankton in vs.plankton_types}

            # for plankton, sat_const in saturation_constants.items():
            for plankton in vs.plankton_types:
                """
                growth rate of phytoplankton
                consume DOP when it is more efficient
                """
                sat_const = saturation_constants[plankton]
                lim_po4 = tracers["po4"] / (sat_const / vs.redfield_ratio_PN  + tracers["po4"])

                if vs.enable_nitrogen:
                    lim_dop = vs.hdop * tracers["DOP"] / (sat_const / vs.redfield_ratio_PN + tracers["DOP"])
                    dopupt_flag = lim_dop > lim_po4
                    limP = lim_dop * dopupt_flag + lim_po4 * np.logical_not(dopupt_flag)
                else:
                    limP = lim_po4

                if vs.enable_nitrogen:
                    lim_no3 = tracers["no3"] / (sat_const + tracers["no3"])

                if plankton == "diazotroph":
                    u = np.minimum(limP, lim_no3) * jmax[plankton]
                else:
                    u = limP * jmax[plankton]

                if vs.enable_iron:
                    u *= iron_limitation(vs, *iron_limits[plankton], tracers["fe"], tracers[plankton], pmax[plankton])

                u = np.minimum(u, avej[plankton])

                net_primary_production[plankton] = u * tracers[plankton]

                if vs.enable_nitrogen:
                    dop_uptake[plankton] = net_primary_production[plankton] * dopupt_flag

                    net_primary_production[plankton] *= (flags["DOP"] * dopupt_flag + flags["po4"] * np.logical_not(dopupt_flag)) * flags["no3"]

                else:
                    net_primary_production[plankton] *= flags["po4"]

                if plankton == "diazotroph":
                    no3_uptake_D = (.5 + .5 * np.tanh(tracers["no3"] - 5.)) * net_primary_production[plankton]  # nitrate uptake TODO: remove magic number 5


            # Recycling of plankton, detritus etc. microbial loop
            # Everything which is turned to nutrients at a fixed rate
            recycled = {recycler: flags[recycler] * rate * tracers[recycler]
                        for recycler, rate in vs.recycling_rates.items()}

            # Michaelis-Menten denominator
            thetaZ = sum([pref_score * tracers[preference] for preference, pref_score in vs.zooplankton_preferences.items()]) + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN #\
#                    + tracers["ballast"] * vs.zooplankton_preferences["detritus"]

            # Ingestion by zooplankton based on preference
            ingestion = {preference: pref_score / thetaZ for preference, pref_score in vs.zooplankton_preferences.items()}

            if vs.enable_calcifiers:
                ingestion["ballast"] = vs.zooplankton_preferences["detritus"] / thetaZ

            # Grazing is based on availability of food and eaters??
            grazing = {preference: flags[preference] * flags["zooplankton"] * gmax *
                       ingestion[preference] * tracers[preference] * tracers["zooplankton"]
                       for preference in vs.zooplankton_preferences}

            if vs.enable_calcifiers:
                grazing["ballast"] = gmax * flags["ballast"] * flags["zooplankton"] * ingestion["ballast"] * tracers["zooplankton"]

            mortality = {plankton: flags[plankton] * rate * tracers[plankton]
                         for plankton, rate in vs.mortality_rates.items()}
            mortality["zooplankton"] *= tracers["zooplankton"]  # quadric rate


            """
            Grazing is split into several parts:
            Digestion: How much of the eaten material is put into growing the population
            Excretion: Eaten metrial is disposed as nutrients. This amount is not available for growing
            Sloppy feeding: Material is not consumed by zooplankton and is lost as detritus
            """

            digestion = {preference: vs.assimilation_efficiency * amount_grazed for preference, amount_grazed in grazing.items()}

            if vs.enable_nitrogen:
                digestion["diazotroph"] *= 1. / (vs.redfield_ratio_PN * vs.diazotroph_NP)

            digestion_total = sum(digestion.values())

            excretion = {preference: vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

            if vs.enable_nitrogen:
                excretion["diazotroph"] *= (vs.redfield_ratio_PN * vs.diazotroph_NP)

                # FIXME is it just me, or is there too much math here?
                nr_excr_D = vs.assimilation_efficiency * grazing["diazotroph"] * (1 - 1./(vs.redfield_ratio_PN * vs.diazotroph_NP)) + (1 - vs.assimilation_efficiency) * grazing["diazotroph"] * (1 - 1./(vs.redfield_ratio_PN * vs.diazotroph_NP))

            excretion_total = sum(excretion.values())

            sloppy_feeding = {preference: (1 - vs.assimilation_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

            if vs.enable_nitrogen:
                sloppy_feeding["diazotroph"] *= 1. / (vs.redfield_ratio_PN * vs.diazotroph_NP)

            sloppy_feeding_total = sum(sloppy_feeding.values())


            if vs.enable_calcifiers:
                dissl[:, :, k] = tracers["caco3"] * dissk1
                remi_ballast = np.maximum(vs.bapr * dissl[:, :, k] / (vs.capr + vs.redfield_ratio_CN * 1e3), 0)
                # net formation of attached (living tests
                # total primary production by coccolitophores + total growth of zooplankton
                calatt[:, :, k] = (net_primary_production["coccolitophore"] - mortality["coccolitophore"] - recycled["coccolitophore"] - grazing["coccolitophore"] + digestion_total - mortality["zooplankton"] - grazing["zooplankton"] - excretion_total) * vs.capr * vs.redfield_ratio_CP * vs.redfield_ratio_PN * 1e3

                calpro[:, :, k] = (sloppy_feeding["coccolitophore"] + sloppy_feeding["zooplankton"] + mortality["coccolitophore"] + mortality["zooplankton"]) * vs.capr * vs.redfield_ratio_CP * vs.redfield_ratio_PN * 1e3

            if vs.enable_iron:
                # par = 0.43 # fraction of photosynthetically active radiation
                # irrtop = grid_light / 2 / par
                # kirr = - light_attenuation
                # aveirr = -1 / vs.dzt / kirr * (irrtop - irrtop * np.exp(kirr * vs.dzt))
                # remife = vs.recycling_rates["particulate_fe"] * tracers["particulate_fe"] * flags["particulate_fe"]
                # Scavenging of dissolved iron is based on Honeymoon et al. (1988)
                # and Parekh et al. (2004).
                flags["o2"] = tracers["o2"] > vs.o2min
                fepa = (1.0 - vs.kfeleq * (vs.lig - tracers["fe"])) * flags["o2"]
                feprime = ((-fepa + (fepa * fepa + 4.0 * vs.kfeleq * tracers["fe"]) ** 0.5) / (2.0 * vs.kfeleq)) * flags["o2"]
                feorgads = (vs.kfeorg * (((tracers["detritus"] * flags["detritus"]) * vs.mc * vs.redfield_ratio_PN * vs.redfield_ratio_CP)**0.58) * feprime) * flags["o2"] * flags["particulate_fe"]
                fecol = vs.kfecol * feprime * flags["o2"] * flags["fe"]


            # Export of all sinking stuff
            tmp_expo = {sinker: speed[:, :, k] * tracers[sinker] * flags[sinker] for sinker, speed in vs.sinking_speeds.items()}


            dtracer = {tracer: np.zeros_like(tracer_dim) for tracer, tracer_dim in tracers.items()}

            nutrients_total = {nutrient: np.zeros_like(tracers[nutrient]) for nutrient in nutrient_factors}


            """
            Model dynamics:
            """

            for plankton, npp in net_primary_production.items():
                dtracer[plankton][:, :] += npp
                dtracer["po4"] += dop_uptake[plankton]

                if vs.enable_nitrogen:
                    dtracer["DOP"] -= dop_uptake[plankton]

                for nutrient in nutrient_factors:
                    nutrients_total[nutrient][:, :] -= npp


            for plankton, mort in mortality.items():
                dtracer[plankton][:, :] += mort
                dtracer["detritus"][:, :] += (1 - uptake_factors[plankton]) * mort

                for dom in doms:
                    dtracer[dom][:, :] += uptake_factors[plankton] * mort

            for grazed, val in grazing.items():
                dtracer[grazed][:, :] -= val

            for recycler, val in recycled.items():
                dtracer[recycler][:, :] -= val

                val_tmp = vs.dfr * val

                for nutrient in nutrient_factors:
                    nutrients_total[nutrient][:, :] += val - val_tmp

                for dom in doms:
                    dtracer[dom] += val_tmp

            for sinker in tmp_expo:
                dtracer[sinker] += impo[sinker] - tmp_expo[sinker]

            for nutrient, factor in nutrient_factors.items():
                dtracer[nutrient][:, :] += factor * (nutrients_total[nutrient] + excretion_total)

            dtracer["zooplankton"][:, :] += digestion_total - excretion_total
            dtracer["detritus"][:, :] += sloppy_feeding_total

            if vs.enable_nitrogen:
                dtracer["no3"][:, :] += nr_excr_D - no3_uptake_D
                dtracer["no3"][:, :] *= (1 - 1. /(vs.redfield_ratio_PN * vs.diazotroph_NP))

            if vs.enable_calcifiers:
                dtracer["caco3"][:, :] += calpro[:, :, k] - dissl[:, :, k]

                # TODO ballast is calculated directly from detritus, no reason for this extra
                dtracer["ballast"][:, :] += vs.bapr * ((1 - vs.dfr) * mortality["phytoplankton"]
                                + sloppy_feeding_total + mortality["zooplankton"]) - remi_ballast \
                                - grazing["ballast"] + vs.bapr \
                                * mortality["diazotroph"] * (vs.redfield_ratio_PN / vs.diazotroph_NP) \
                                + vs.bapr * (1 - vs.dfr) * mortality["coccolitophore"]

                dtracer["DIC"][:, :] += excretion_total + recycled["DON"] + nr_excr_D \
                            + mortality["diazotroph"] * (1 - 1 / (vs.redfield_ratio_PN / vs.diazotroph_NP)
)
                dtracer["DIC"][:, :] *= vs.redfield_ratio_CN

            if vs.enable_iron:
                dtracer["fe"][:, :] += - feorgads - fecol + recycled["DON"] + nr_excr_D# + remife

                # TODO should be calculated directly from detritus?
                dtracer["particulate_fe"][:, :] += vs.rfeton * (sloppy_feeding_total + (1 - vs.dfr) * mortality["phytoplankton"] + mortality["diazotroph"] * (vs.redfield_ratio_PN / vs.diazotroph_NP) + mortality["zooplankton"] - grazing["detritus"])
                dtracer["particulate_fe"][:, :] += feorgads + fecol - vs.rfeton * grazing["ballast"]# - remife

            for tracer, val in dtracer.items():
                tracers[tracer][:, :] += vs.dt_bio * val


            """
            End model dynamics
            """

            # update export from layer and flags
            for sinker in vs.sinking_speeds:
                export[sinker][:, :] += tmp_expo[sinker] * vs.dt_bio

            if vs.enable_nitrogen:
                nfix[:, :, k] += net_primary_production["diazotroph"] - no3_uptake_D

            update_flags_and_tracers(vs, flags, tracers)


        # !-----------------------------------------------------------------------
        # !        benthic denitrification model of Bohlen et al., 2012, GBC (eq. 10)
        # !        NO3 is removed out of bottom water nitrate.
        # !        See Somes et al., 2012, BGS for additional details/results
        # !-----------------------------------------------------------------------
        remibot = (vs.kbot - 1 == k) * export["detritus"]

        # # Instantly remineralize at the bottom
        tracers["po4"] += remibot * vs.redfield_ratio_PN


        if vs.enable_nitrogen:
        #     # TODO remove magic numbers
        #     # TODO what do these things mean?
            sg_bdeni = (0.06 + 0.19 * 0.99**(tracers["o2"] - tracers["no3"])) * vs.redfield_ratio_CN * 1e3
            expo = export["detritus"] + export["ballast"] if vs.enable_calcifiers else 0
            sg_bdeni *= expo
            sg_bdeni = np.minimum(sg_bdeni, expo)
            sg_bdeni = np.maximum(sg_bdeni, 0)
            sg_bdeni *= (0.5 + 0.5 * np.tanh(tracers["no3"] * 10 - 5.0)) * flags["no3"]
            bdeni[:, :, k] = sg_bdeni[:, :]

        #     # Instantly remineralize at the bottom
            tracers["no3"] += remibot - sg_bdeni

        if vs.enable_iron:
            # Something about sediment sources of iron
            fesed = vs.fetopsed * bct * vs.redfield_ratio_PN * remibot
            tracers["fe"] += fesed

            # completely remineralize iron at bottom, if enough available oxygen
            remibotfe = flags["o2"] * (vs.kbot - 1 == k) * export["particulate_fe"]
            tracers["fe"] += remibotfe
            export["particulate_fe"] -= remibotfe

        if vs.enable_calcifiers:
            tracers["DIC"] += remibot * vs.redfield_ratio_CN
            tracers["alkalinity"] += -tracers["DIC"] * vs.redfield_ratio_NC * 1e-3

        export["detritus"] -= remibot

        # # Calculate total export to get total import for next layer
        for sinker in vs.sinking_speeds:
            export[sinker][:, :] *= vs.dzt[k] / nbio

        # # Update model values
        for name, value in tracers.items():
            vs.npzd_tracers[name][:, :, k] = value

    if vs.enable_calcifiers:
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
            so2 = vs.npzd_tracers["DIC"][:, :, k] * vs.redfield_ratio_OC + nfix[:, :, k] * (1.25e-3 / nbio)
            flags["no3"] = vs.npzd_tracers["no3"] > vs.trcmin  # TODO why would this check be necessary?
            limit_no3 = .5 * np.tanh(vs.npzd_tracers["no3"][:, :, k] - 2.5)
            # limit_ntp = 0.5 * np.tanh(vs.npzd_tracers["no3"] / (vs.refield_ratio_NP * vs.npzd_tracers["po4"]) * 100 - 60.)
            # 800 = (elec/mol O2)/(elec/mol NO3)*(mmol/mol)
            wcdeni = 800 * flags["no3"][:, :, k] * so2 * (.5 - fo2) * (.5 * limit_no3)
            wcdeni = np.maximum(wcdeni, 0)
            vs.npzd_tracers["no3"][:, :, k] -= wcdeni

            vs.npzd_tracers["alkalinity"][:, :, k] += wcdeni * 1e-3 + bdeni[:, :, k] * 1e-3 - nfix[:, :, k] / nbio * 1e-3

        for k in range(vs.nz - 1):
            vs.npzd_tracers["DIC"][:, :, k] += dissl[:, :, k] / nbio * 1e-3 - calpro[:, :, k] / nbio - calatt[:, :, k] * 1e-3
            vs.npzd_tracers["alkalinity"][:, :, k] += 2 * dissl[:, :, k] / nbio * 1e-3 - 2 * calpro[:, :, k] / nbio - 2 * calatt[:, :, k] * 1e-3

        vs.npzd_tracers["DIC"][:, :, vs.nz - 1] += dissl[:, :, vs.nz - 1] / nbio * 1e-3 - calpro[:, :, vs.nz - 1] / nbio * 1e-3 - calatt[:, :, vs.nz - 1] / nbio * 1e-3 + export["caco3"] * 1e-3
        vs.npzd_tracers["alkalinity"][:, :, vs.nz - 1] += 2 * dissl[:, :, vs.nz - 1] / nbio * 1e-3 - 2 * calpro[:, :, vs.nz - 1] / nbio * 1e-3 - 2 * calatt[:, :, vs.nz - 1] / nbio * 1e-3 + 2 * export["caco3"] * 1e-3




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
        tracers[key] = np.maximum(tracers[key], vs.trcmin)

@veros_method
def iron_limitation(vs, k_min, k_max, iron, plankton, pmax):
    """Calculate limiting max growth due to iron defficiency"""
    p1 = np.minimum(plankton, pmax)
    p2 = np.maximum(0, plankton - pmax)

    k = (k_min * p1 + k_max * p2) / (p1 + p2)

    limFe = iron / (k + iron)
    return limFe


@veros_method
def npzd(vs):
    """
    Main driving function for NPZD functionality
    Computes transport terms and biological activity separately

    $$
    \dfrac{\partial C_i}{\partial t} = T + S
    $$
    """

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
