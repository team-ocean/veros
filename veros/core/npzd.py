"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
import numpy as np  # NOTE np is already defined somehow
from .. import veros_method
from . import advection, diffusion, thermodynamics, cyclic, atmospherefluxes, utilities


@veros_method
def biogeochemistry(vs):
    """
    Integrate biochemistry: phytoplankton, zooplankton, detritus, po4
    """

    # Number of timesteps to do for bio tracers
    nbio = int(vs.dt_mom // vs.dt_bio)

    # Integrated phytplankton
    phyto_integrated = np.zeros_like(vs.phytoplankton[:, :, 0])


    # TODO: Make this more elegant...
    # Nutrient exchange at surface
    surfaces_fluxes = [rule[0](vs, rule[1], rule[2]) for rule in vs.nutrient_surface_fluxes]
    for flux in surfaces_fluxes:
        for key, value in flux.items():
            # scale flux by timestep size and depth
            # we get input in mmol/m^2/s and want to convert to mmol/m^3
            vs.npzd_tracers[key][:, :, -1] += value * vs.dt_mom / vs.dzt[-1]
            # vs.npzd_tracers[key][vs.maskT[:, :, -1].astype(bool), -1] += value[vs.maskT[:, :, -1].astype(bool)] * (vs.dt_mom / vs.dzt[-1])

    # import from layer above, export to layer below
    # TODO reset rather than create - reset might not even be necessary
    impo = {sinker: np.zeros((vs.detritus.shape[:2])) for sinker in vs.sinking_speeds}
    export = {sinker: np.zeros_like(imp) for sinker, imp in impo.items()}

    swr = vs.swr.copy()

    # recycling rate determined according to b ** (cT)
    # zooplankton growth rate maxes out at 20C, TODO remove magic number
    bct = vs.bbio ** (vs.cbio * vs.temp[:, :, :, vs.taum1])
    bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, :, vs.taum1], 20))

    # Maximum grazing rate is a function of temperature
    # bctz sets an upper limit on effects of temperature on grazing
    gmax = vs.gbio * bctz

    if vs.enable_carbon:
    #     # reset
    #     vs.prca[:, :] = 0
        vs.calpro = np.zeros_like(vs.prca) # TODO: Just reset?


    # TODO these could be created elsewhere
    # TODO make these local somehow, so they don't pollute the variable set
    vs.net_primary_production = {}
    vs.recycled = {}
    vs.mortality = {}

    for k in reversed(range(vs.nz)):

        # temporary storage of tracers to be modified
        tracers = {name: value[:, :, k] for name, value in vs.npzd_tracers.items()}

        # flags to prevent more outgoing flux than available capacity - stability?
        flags = {tracer: True for tracer in tracers}

        # Set flags and tracers based on minimum concentration
        update_flags_and_tracers(vs, flags, tracers, k, refresh=True)

        # incomming radiation at layer
        swr[...] *= np.exp(- vs.light_attenuation_phytoplankton * phyto_integrated)


        # TODO remove if statements in running code
        # if vs.enable_calcifiers:
        #     swr[:, :] *= np.exp(- vs.light_attenuation_caco3 * tracers["caco3"])
        #     dissk1 = vs.dissk0 * (1 - vs.Omega_c)
        #     # caco3_integrated[...] += np.maximum(tracers["caco3"], vs.trcmin) * vs.dzt[k]

        # integrated phytoplankton for use in next layer
        # TODO: Again just reset?
        phyto_integrated = sum([np.maximum(tracers[plankton], vs.trcmin) for plankton in vs.plankton_types]) * vs.dzt[k]


        ## TODO: Update nud, bctz based on oxygen availability

        # light at top of gridbox
        grid_light = swr * np.exp(vs.zw[k] * vs.rctheta)

        # calculate import of sinking particles from layer above
        for sinker in vs.sinking_speeds:
            impo[sinker] = export[sinker] / (vs.dzt[k] * vs.dt_bio)
            export[sinker][:, :] = 0


        # How much light gets through to this level
        light_attenuation = vs.dzt[k] * (vs.light_attenuation_water
                          + vs.light_attenuation_phytoplankton
                          * sum([tracers[plankton] for plankton in vs.plankton_types]))

        # if vs.enable_calcifiers:
        #     light_attenuation += vs.dzt[k] * vs.light_attenuation_caco3 * tracers["caco3"]

        # calculate maxing growth functions and averaged? growth
        jmax, avej = {}, {}
        for plankton, growth_function in vs.plankton_growth_functions.items():
            jmax[plankton], avej[plankton] = growth_function(vs, k, bct, grid_light, light_attenuation)


        # bio loop
        for _ in range(nbio):

            # Plankton is recycled, dying and growing
            for plankton in vs.plankton_types:

                # Nutrient limiting growth
                u = np.inf

                # TODO Create "rule factory" to turn this into single rules used below
                for growth_limiting_function in vs.limiting_functions[plankton]:
                    u = np.minimum(u, growth_limiting_function(vs, tracers))

                vs.net_primary_production[plankton] = np.minimum(avej[plankton], u * jmax[plankton])

                # Fast recycling of plankton
                # TODO Could this be moved to individual rules?
                vs.recycled[plankton] = flags[plankton] * vs.nupt0 * bct[:, :, k] * tracers[plankton]  # TODO check nupt0 for other plankton types

                # Mortality of plankton
                # Would probably be easier to handle as rules
                vs.mortality[plankton] = flags[plankton] * vs.specific_mortality_phytoplankton * tracers[plankton]  # TODO proper mortality rates for other plankton

            # Detritus is recycled
            vs.recycled["detritus"] = flags["detritus"] * vs.nud0 * bct[:, :, k] * tracers["detritus"]

            # zooplankton displays quadric mortality rates
            vs.mortality["zooplankton"] = flags["zooplankton"] * vs.quadric_mortality_zooplankton * tracers["zooplankton"] ** 2

            # TODO: move these to rules except grazing
            vs.grazing, vs.digestion, vs.excretion, vs.sloppy_feeding = zooplankton_grazing(vs, tracers, flags, gmax[:, :, k])
            vs.excretion_total = sum(vs.excretion.values())  # TODO this one should not be necessary with the rules


            # Temporary export for use in next layer
            tmp_expo = {sinker: speed[:, :, k] * tracers[sinker] * flags[sinker] for sinker, speed in vs.sinking_speeds.items()}

            # Gather all state updates
            npzd_updates = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_rules]

            # perform updates
            for update in npzd_updates:
                for key, value in update.items():
                    vs.npzd_tracers[key][:, :, k] += value * vs.dt_bio  # NOTE far too many multiplications with dt
#                    tracers[key][:, :, k] += value * vs.dt_bio  # NOTE far too many multiplications with dt
#                    vs.npzd_tracers[key][vs.maskT[:, :, k].astype(bool), k] += value[vs.maskT[:, :, k].astype(bool)] * vs.dt_bio

            # Import and export between layers
            for tracer in vs.sinking_speeds:
                vs.npzd_tracers[tracer][:, :, k] += impo[tracer] - tmp_expo[tracer]
                # tracers[tracer][:, :, k] += impo[tracer] - tmp_expo[tracer]

                # Calculate total export from layer to next
                export[tracer][:, :] += tmp_expo[tracer] * vs.dt_bio

            # Prepare temporary tracers for next bio iteration
            # NOTE this doesn't reset the tracer, because vs.npzd_tracers
            # have been updated - it simply gets the update
            # TODO: Is it possible to work directly on npzd_tracers in stead?
            for name in vs.npzd_tracers:
                tracers[name] = vs.npzd_tracers[name][:, :, k]

            # reset flags and tracers
            update_flags_and_tracers(vs, flags, tracers, k)

        # Remineralize at bottom, and ensure sinking stuff doesn't fall through
        # at_bottom = (vs.kbot - 1) == k
        at_bottom = (vs.kbot == k + 1) # TODO only calculate this once
        tracers["po4"][at_bottom] += export["detritus"][at_bottom] * vs.redfield_ratio_PN

        if vs.enable_carbon:  # TODO remove ifs - maybe create common nutrient rule
            tracers["DIC"][at_bottom] += export["detritus"][at_bottom] * vs.redfield_ratio_CN

        for sinker in export:
            export[sinker][:, :] -= at_bottom * export[sinker]

        # Calculate export for next layer
        # for sinker in export:
            export[sinker][:, :] *= vs.dzt[k] / nbio

        if vs.enable_carbon:  # TODO remove ifs in process code
            # dprca = export["caco3"] * 1e-3
            dprca = vs.calpro / nbio #* 1e-3
            vs.prca += dprca * vs.dzt[k]
            tracers["DIC"][:, :] -= dprca

            vs.alkalinity[:, :, k] = - tracers["DIC"] / vs.redfield_ratio_CN# * 1e-3

            if not vs.enable_calcifiers:
                vs.alkalinity[:, :, k] -= 2 * dprca

            vs.calpro[:, :] = 0

        for name, value in tracers.items():
            vs.npzd_tracers[name][:, :, k] = value

    # Remineralization of calcite - NOTE: Why can't this be in the main loop?
    if vs.enable_carbon:

        # TODO shouldn't rcab be at the bottom? rather than at the very last
        vs.npzd_tracers["DIC"][:, :, 1:] += vs.prca[:, :, np.newaxis] * vs.rcak[:, :, 1:]
        vs.npzd_tracers["DIC"][:, :, 0] += vs.prca * vs.rcab[:, :, 0]

        vs.alkalinity[:, :, 1:] += 2 * vs.prca[:, :, np.newaxis] * vs.rcak[:, :, 1:]
        vs.alkalinity[:, :, 0] += 2 * vs.prca * vs.rcab[:, :, 0]
        vs.prca[:, :] = 0


# TODO additional k-loops

@veros_method
def zooplankton_grazing(vs, tracers, flags, gmax):
    zprefs = vs.zprefs

    thetaZ = sum([pref_score * tracers[preference] for preference, pref_score in zprefs.items()]) + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

    ingestion = {preference: pref_score / thetaZ for preference, pref_score in zprefs.items()}

    grazing = {preference: flags[preference] * flags["zooplankton"] * gmax * ingestion[preference] * tracers[preference] * tracers["zooplankton"] for preference in ingestion}

    digestion = {preference: vs.assimilation_efficiency * amount_grazed for preference, amount_grazed in grazing.items()}

    excretion = {preference: vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

    sloppy_feeding = {preference: (1 - vs.assimilation_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

    return grazing, digestion, excretion, sloppy_feeding


@veros_method
def potential_growth(vs, k, bct, grid_light, light_attenuation, growth_parameter):
    f1 = np.exp(-light_attenuation)
    jmax = growth_parameter * bct[:, :, k]
    gd = jmax * vs.dt_mom
    avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

    return jmax, avej

@veros_method
def phytoplankton_potential_growth(vs, k, bct, grid_light, light_attenuation):
    return potential_growth(vs, k, bct, grid_light, light_attenuation, vs.abio_P)

@veros_method
def coccolitophore_potential_growth(vs, k, bct, grid_light, light_attenuation):
    return potential_growth(vs, k, bct, grid_light, light_attenuation, vs.abio_C)


@veros_method
def diazotroph_potential_growth(vs, k, bct, grid_light, light_attenuation):
    f1 = np.exp(-light_attenuation)
    jmax = np.maximum(0, vs.abio_P * vs.jdiar * (bct[:, :, k] - 2.6))
    gd = np.maximum(1e-14, jmax * vs.dt_mom)
    avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

    return jmax, avej


# @veros_method
# def calc_net_primary_production(vs, tracers, flags, jmax, avej, saturation_constant, plankton):
#     limit_phosphate = tracers["po4"] / (saturation_constant + tracers["po4"])
#     u = limit_phosphate * jmax
#     u = np.minimum(u, avej)
#     net_primary_production = u * tracers[plankton] * flags["po4"]

#     return net_primary_production

@veros_method
def avg_J(vs, f1, gd, grid_light, light_attenuation):
    """Average J"""
    u1 = np.maximum(grid_light / gd, 1e-6)  # TODO remove magic number 1e-6
    u2 = u1 * f1

    # NOTE: There is an approximation here: u1 < 20 WTF? Why 20?
    phi1 = np.log(u1 + np.sqrt(1 + u1**2)) - (np.sqrt(1 + u1**2) - 1) / u1
    phi2 = np.log(u2 + np.sqrt(1 + u2**2)) - (np.sqrt(1 + u2**2) - 1) / u2



    return gd * (phi1 - phi2) / light_attenuation

def general_nutrient_limitation(nutrient, saturation_constant):
    return nutrient / (saturation_constant + nutrient)

@veros_method
def phosphate_limitation_phytoplankton(vs, tracers):
    return general_nutrient_limitation(tracers["po4"], vs.saturation_constant_N / vs.redfield_ratio_PN)

@veros_method
def phosphate_limitation_coccolitophore(vs, tracers):
    return general_nutrient_limitation(tracers["po4"], vs.saturation_constant_NC / vs.redfield_ratio_PN)

@veros_method
def phosphate_limitation_diazotroph(vs, tracers):
    return general_nutrient_limitation(tracers["po4"], vs.saturation_constant_N / vs.redfield_ratio_PN)

@veros_method
def nitrate_limitation_diazotroph(vs, tracers):
    return general_nutrient_limitation(tracers["no3"], vs.saturation_constant_N)

@veros_method
def update_flags_and_tracers(vs, flags, tracers, k, refresh=False):
    """Set flags"""

    for key in tracers:
        keep = flags[key] if not refresh else True  # if the flag was already false, keep it false
        flags[key] = ((tracers[key] * vs.maskT[:, :, k]) > vs.trcmin) * keep
        tracers[key] = np.maximum(tracers[key] * vs.maskT[:, :, k], vs.trcmin)

@veros_method
def register_npzd_data(vs, name, value=0):

    if name not in vs.npzd_tracers:
        if vs.show_npzd_graph:
            vs.npzd_graph.attr('node', shape='square')
            vs.npzd_graph.node(name)
    else:
        print(name, "has already been added to the NPZD data set, value has been updated")

    vs.npzd_tracers[name] = value

@veros_method
def register_npzd_rule(vs, function, source, destination, label="?"):
    """ Add rule to the npzd dynamics e.g. phytoplankkton being eaten by zooplankton

        ...
        function: function to be called
        source: what is being consumed
        destination: what is growing from consuming
        label: A description for graph
        ...
    """
    vs.npzd_rules.append((function, source, destination))
    if vs.show_npzd_graph:
        #vs.npzd_graph.edge(source, destination, label=label)
        vs.npzd_graph.edge(source, destination, label=label, lblstyle="above, sloped")
        # it is also possible to add tooltiplabels for more explanation
        # TODO would it be possible to add lblstyle to work with tikz/pgf?

@veros_method
def empty_rule(*args):
    """ An empty rule for providing structure"""
    return {}

@veros_method
def primary_production(vs, nutrient, plankton):
    """ Primary production: Growth by consumption of light and nutrients """
    return {nutrient: - vs.redfield_ratio_PN * vs.net_primary_production[plankton], plankton: vs.net_primary_production[plankton]}

@veros_method
def recycling(vs, plankton, nutrient, ratio):
    """ plankton or detritus is recycled into nutrients """
    return {nutrient: ratio * vs.recycled[plankton], plankton: - vs.recycled[plankton]}

@veros_method
def mortality(vs, plankton, detritus):
    """ All dead matter from plankton is converted to detritus """
    return {plankton: - vs.mortality[plankton], detritus: vs.mortality[plankton]}

@veros_method
def sloppy_feeding(vs, plankton, detritus):
    """ When zooplankton graces, some is not eaten. This is converted to detritus.
        There should be a rule for sloppy feeding for each grazing"""
    return {detritus: vs.sloppy_feeding[plankton]}

@veros_method
def grazing(vs, eaten, zooplankton):
    """ Zooplankton grows by amount digested, eaten decreases by amount eaten """
    return {eaten: - vs.grazing[eaten], zooplankton: vs.digestion[eaten]}

@veros_method
def zooplankton_self_grazing(vs, zooplankton1, zoooplankton2):
    """ Zooplankton grazing on itself doesn't work with the grazing function
        because it would overwrite dictionary keys
        therefore we implement a special rule for them
        zooplankton2 is superflous, but necessary for the code to run
    """
    return {zooplankton1: vs.digestion[zooplankton1] - vs.grazing[zooplankton1]}

@veros_method
def excretion(vs, zooplankton, nutrient):
    """ Zooplankton excretes nutrients after eating. Poop, breathing... """
    return {zooplankton: - vs.excretion_total, nutrient: vs.redfield_ratio_PN * vs.excretion_total}

@veros_method
def primary_production_from_DIC(vs, nutrient, plankton):
    """ Only using DIC, because plankton is handled by po4 ... shitty design right now """
    # print( (vs.redfield_ratio_CN * vs.net_primary_production[plankton]).mean())
    return {nutrient: - vs.redfield_ratio_CN * vs.net_primary_production[plankton]}

@veros_method
def recycling_to_po4(vs, plankton, phosphate):
    return recycling(vs, plankton, phosphate, vs.redfield_ratio_PN)

@veros_method
def recycling_to_no3(vs, plankton, no3):
    return recycling(vs, plankton, no3, 1)

@veros_method
def recycling_to_dic(vs, plankton, dic):
    return recycling(vs, plankton, dic, vs.redfield_ratio_CN)

@veros_method
def recycling_phyto_to_dic(vs, plankton, dic):
    return recycling(vs, plankton, dic, (1 - vs.dfrt) * vs.redfield_ratio_CN)

@veros_method
def excretion_dic(vs, zooplankton, nutrient):
    """ Zooplankton excretes nutrients after eating. Poop, breathing... """
    #return {zooplankton: - vs.excretion_total, nutrient: vs.redfield_ratio_CN * vs.excretion_total}
    return {nutrient: vs.redfield_ratio_CN * vs.excretion_total}

@veros_method
def calpro(vs, plankton, cal):
    # return {cal: (vs.mortality[plankton] + vs.sloppy_feeding[plankton]) * vs.capr * vs.redfield_ratio_CN * 1e3}
    vs.calpro += (vs.mortality[plankton] + vs.grazing[plankton] * (1 - vs.assimilation_efficiency)) * vs.capr * vs.redfield_ratio_CN #* 1.e3
    # return {cal: vs.calpro}
    return {}

@veros_method
def co2_surface_flux(vs, co2, dic):
    # TODO not global please
    vs.cflux[...] = atmospherefluxes.carbon_flux(vs)
    return {dic: vs.cflux}

@veros_method
def setupNPZD(vs):
    """Taking veros variables and packaging them up into iterables"""
    vs.npzd_tracers = {}  # Dictionary keeping track of plankton, nutrients etc.
    vs.npzd_rules = []  # List of rules describing the interaction between tracers
    vs.nutrient_surface_fluxes = []
    if vs.show_npzd_graph:
        from graphviz import Digraph
        vs.npzd_graph = Digraph("npzd_dynamics", filename="npzd_dynamics.gv", format="png")  # graph for visualizing interactions - usefull for debugging
    vs.plankton_growth_functions = {}  # Dictionary containing functions describing growth of plankton
    vs.limiting_functions = {}  # Contains descriptions of how nutrients put a limit on growth

    vs.sinking_speeds = {}  # Dictionary of sinking objects with their sinking speeds
    vs.sinking_speeds["detritus"] = (vs.wd0 + vs.mw * np.where(-vs.zw < vs.mwz, -vs.zw, vs.mwz)) / vs.dzt * vs.maskT
    # TODO: What is the reason for using zw rather than zt?

    # Add "regular" phytoplankton to the model
    vs.plankton_types = ["phytoplankton"]  # Phytoplankton types in the model. Used for blocking ligh + ..
    vs.plankton_growth_functions["phytoplankton"] = phytoplankton_potential_growth
    vs.limiting_functions["phytoplankton"] = [phosphate_limitation_phytoplankton]

    # Zooplankton preferences for grazing on keys
    # Values are scaled automatically at the end of this function
    vs.zprefs = {"phytoplankton": vs.zprefP, "zooplankton": vs.zprefZ, "detritus": vs.zprefDet}

    # Register for basic model
    register_npzd_data(vs, "phytoplankton", vs.phytoplankton)
    register_npzd_data(vs, "zooplankton", vs.zooplankton)
    register_npzd_data(vs, "detritus", vs.detritus)
    register_npzd_data(vs, "po4", vs.po4)

    # Describe interactions between elements in model
    # function describing interaction, from, to, description for graph
    register_npzd_rule(vs, grazing, "phytoplankton", "zooplankton", label="Grazing")
    register_npzd_rule(vs, mortality, "phytoplankton", "detritus", label="Mortality")
    register_npzd_rule(vs, sloppy_feeding, "phytoplankton", "detritus", label="Sloppy feeding")
    register_npzd_rule(vs, recycling_to_po4, "phytoplankton", "po4", label="Fast recycling")
    register_npzd_rule(vs, zooplankton_self_grazing, "zooplankton", "zooplankton", label="Grazing")
    register_npzd_rule(vs, excretion, "zooplankton", "po4", label="Excretion")
    register_npzd_rule(vs, mortality, "zooplankton", "detritus", label="Mortality")
    register_npzd_rule(vs, sloppy_feeding, "zooplankton", "detritus", label="Sloppy feeding")
    register_npzd_rule(vs, sloppy_feeding, "detritus", "detritus", label="Sloppy feeding")
    register_npzd_rule(vs, grazing, "detritus", "zooplankton", label="Grazing")
    register_npzd_rule(vs, recycling_to_po4, "detritus", "po4", label="Remineralization")
    register_npzd_rule(vs, primary_production, "po4", "phytoplankton", label="Primary production")


    # Add carbon to the model
    # TODO complete interactions
    if vs.enable_carbon:
        # Preparation of some variables for remineralization of caco3
        vs.prca = np.zeros_like(vs.dic[:, :, 0])
      # dcaco3 = 650000.0  ! remineralisation depth of calcite [cm]
        dcaco3 = 65000.0 / 100  # remineralization depth of calcite [m]

        vs.rcak = np.empty_like(vs.dic)
        vs.rcak[:, :, -1] = - (np.exp(vs.zw[-1] / dcaco3) - 1) / vs.dzt[-1]
        vs.rcak[:, :, :-1] = (- np.exp(vs.zw[:-1] / dcaco3) + np.exp(vs.zw[1:] / dcaco3)) / vs.dzt[:-1]
        vs.rcak[...] *= vs.maskT


        vs.rcab = np.empty_like(vs.dic)
        vs.rcab[:, : -1] = 1 / vs.dzt[-1]
        vs.rcab[:, :, :-1] = np.exp(vs.zw[:-1] / dcaco3) / vs.dzt[:-1]
        vs.rcab[...] *= vs.maskT

        # Need to track dissolved inorganic carbon, alkalinity and calcium carbonate
        register_npzd_data(vs, "DIC", vs.dic)
        vs.nutrient_surface_fluxes.append((co2_surface_flux, "co2", "DIC"))

        register_npzd_rule(vs, recycling_to_dic, "detritus", "DIC", label="Remineralization")
        register_npzd_rule(vs, primary_production_from_DIC, "DIC", "phytoplankton", label="Primary production")
        register_npzd_rule(vs, recycling_phyto_to_dic, "phytoplankton", "DIC", label="Fast recycling")
        register_npzd_rule(vs, excretion_dic, "zooplankton", "DIC", label="Excretion")


        vs.hSWS[:, :] = 0.5 * (1e-6 + 1e-10) # for initial guess


        # These rules will be different if we track coccolithophores
        if not vs.enable_calcifiers:
            register_npzd_rule(vs, calpro, "phytoplankton", "caco3", label="Production of calcite")
            register_npzd_rule(vs, calpro, "zooplankton", "caco3", label="Production of calcite")

            # rules below do nothing. They are just here for show..
            register_npzd_rule(vs, empty_rule, "alkalinity", "caco3", label="Production")
            register_npzd_rule(vs, empty_rule, "caco3", "alkalinity", label="Dissolution")


    # Add nitrogen cycling to the model
    # TODO complete rules:
    #       - Primary production rules need to be generalized
    #       - DOP, DON availability needs to be considered
    if vs.enable_nitrogen:
        register_npzd_data(vs, "diazotroph", vs.diazotroph)
        register_npzd_data(vs, "no3", vs.no3)
        register_npzd_data(vs, "DOP", vs.dop)
        register_npzd_data(vs, "DON", vs.don)

        vs.zprefs["diazotroph"] = vs.zprefD  # Add preference for zooplankton to graze on diazotrophs
        vs.plankton_types.append("diazotroph")  # Diazotroph behaces like plankton
        vs.plankton_growth_functions["diazotroph"] = phytoplankton_potential_growth  # growth function
        vs.limiting_functions["diazotroph"] = [phosphate_limitation_diazotroph, nitrate_limitation_diazotroph]  # Limited in nutrients by both phosphate and nitrate

        register_npzd_rule(vs, grazing, "diazotroph", "zooplankton", label="Grazing")
        register_npzd_rule(vs, recycling_to_po4, "diazotroph", "po4", label="Fast recycling")
        register_npzd_rule(vs, recycling_to_no3, "diazotroph", "no3", label="Fast recycling")
        register_npzd_rule(vs, empty_rule, "diazotroph", "DON", label="Fast recycling")
        register_npzd_rule(vs, empty_rule, "diazotroph", "DOP", label="Fast recycling")
        register_npzd_rule(vs, empty_rule, "po4", "diazotroph", label="Primary production")
        register_npzd_rule(vs, empty_rule, "no3", "diazotroph", label="Primary production")
        register_npzd_rule(vs, empty_rule, "DOP", "diazotroph", label="Primary production")
        register_npzd_rule(vs, empty_rule, "DON", "diazotroph", label="Primary production")
        register_npzd_rule(vs, mortality, "diazotroph", "detritus", label="Sloppy feeding")
        register_npzd_rule(vs, recycling_to_no3, "detritus", "no3", label="Remineralization")
        register_npzd_rule(vs, excretion, "zooplankton", "no3", label="Excretion")
        register_npzd_rule(vs, empty_rule, "DOP", "po4", label="Remineralization??")
        register_npzd_rule(vs, empty_rule, "DON", "no3", label="Remineralization??")
        register_npzd_rule(vs, empty_rule, "DOP", "phytoplankton", label="Primary production")


    # Add calcifying "stuff"
    # TODO: complete rules: Should be trivial if nitrogen is working
    if vs.enable_calcifiers:
        # vs.redfield_ratio_CN = vs.redfield_ratio_CP * vs.redfield_ratio_PN
        vs.zprefs["coccolitophore"] = vs.zprefC
        vs.plankton_types.append("coccolitophore")

        # TODO correct growth function
        vs.plankton_growth_functions["coccolitophore"] = coccolitophore_potential_growth
        vs.limiting_functions["coccolitophore"] = [phosphate_limitation_coccolitophore]

        vs.sinking_speeds["caco3"] = (vs.wc0 + vs.mw_c * np.where(-vs.zw < vs.mwz, -vs.zw, vs.mwz)) / vs.dzt * vs.maskT

        register_npzd_data(vs, "caco3", vs.caco3)
        register_npzd_data(vs, "coccolitophore", vs.coccolitophore)

        register_npzd_rule(vs, primary_production, "po4", "coccolitophore", label="Primary production")
        register_npzd_rule(vs, recycling_to_po4, "coccolitophore", "po4", label="Fast recycling")
        register_npzd_rule(vs, mortality, "coccolitophore", "detritus", label="Mortality")
        register_npzd_rule(vs, recycling_phyto_to_dic, "coccolitophore", "DIC", label="Fast recycling")
        register_npzd_rule(vs, primary_production_from_DIC, "DIC", "coccolitophore", label="Primary production")
        register_npzd_rule(vs, grazing, "coccolitophore", "zooplankton", label="Grazing")

        register_npzd_rule(vs, calpro, "coccolitophore", "caco3", label="Calcite production due to sloppy feeding???")
        register_npzd_rule(vs, calpro, "zooplankton", "caco3", label="Calcite production due to sloppy feeding???")


    # TODO add iron back into the model
    if vs.enable_iron:
        pass

    # TODO add oxygen back into the model
    if vs.enable_oxygen:
        pass

    # Update Zooplankton preferences dynamically
    zprefsum = sum(vs.zprefs.values())
    for preference in vs.zprefs:
        vs.zprefs[preference] /= zprefsum

    # Whether or not to display a graph of the intended dynamics to the user
    # TODO move this to diagnostics
    if vs.show_npzd_graph:
        vs.npzd_graph.view()


@veros_method
def npzd(vs):
    """
    Main driving function for NPZD functionality
    Computes transport terms and biological activity separately

    :math: \dfrac{\partial C_i}{\partial t} = T + S
    """

    # For keeping track of changes due to transport
    vs.tracer_result = {}

    for tracer, val in vs.npzd_tracers.items():
        dNPZD_advect = np.zeros_like(val)
        dNPZD_diff = np.zeros_like(val)

        # NOTE Why is this in thermodynamics?
        thermodynamics.advect_tracer(vs, val, dNPZD_advect)
        # vs.tracer_result[tracer] = vs.dt_mom * ((1.5 + vs.AB_eps) * dNPZD_advect - (0.5 + vs.AB_eps) * dNPZD_advect)
        vs.tracer_result[tracer] = vs.dt_mom * dNPZD_advect


        # TODO distinguish between biharmonic mixing and simple diffusion like in thermodynamics

        # diffusion.biharmonic(vs, val, vs.K_hbi, dNPZD_diff)  # TODO correct parameter
        diffusion.biharmonic(vs, val, np.sqrt(abs(vs.K_hbi)), dNPZD_diff)  # TODO correct parameter
        vs.tracer_result[tracer] += vs.dt_mom * dNPZD_diff

        # TODO isopycnal diffusion



        """
        Vertical mixing
        """

        # TODO: Most of this could probably be moved outside
        # dtracer_vmix = val[...]
        # a_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
        # b_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
        # c_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
        # d_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
        # delta = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)

        # ks = vs.kbot[2:-2, 2:-2] - 1
        # delta[:, :, :-1] = vs.dt_tracer / vs.dzw[np.newaxis, np.newaxis, :-1] \
        #     * vs.kappaH[2:-2, 2:-2, :-1]
        # delta[:, :, -1] = 0.
        # a_tri[:, :, 1:] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:]
        # b_tri[:, :, 1:] = 1 + (delta[:, :, 1:] + delta[:, :, :-1]) \
        #     / vs.dzt[np.newaxis, np.newaxis, 1:]
        # b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
        # c_tri[:, :, :-1] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1]



        # d_tri[...] = val[2:-2, 2:-2, :]

        # # TODO use dt_tracer in stead of dt_mom
        # # d_tri[:, :, -1] += vs.dt_mom * vs.forc_temp_surface[2:-2, 2:-2] / vs.dzt[-1]
        # d_tri[:, :, -1] += 0  # NOTE: 0 because no forcing of nutrients or plankton




        # sol, mask = utilities.solve_implicit(vs, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)

        # val[2:-2, 2:-2, :] = utilities.where(vs, mask, sol, val[2:-2, 2:-2])

        # if tracer == "DIC":
        #     vs.ddic_vmix = (val - dtracer_mix) / vs.dt_mom # TODO td_tracer


        # vs.tracer_result[tracer] = vs.dt_mom * (dNPZD_advect + dNPZD_diff) # NOTE: old before Adam-Bashforth etc.


    biogeochemistry(vs)

    for tracer in vs.npzd_tracers:
        vs.npzd_tracers[tracer][...] += vs.tracer_result[tracer]

    for tracer in vs.npzd_tracers.values():
        tracer[...] = np.maximum(tracer, vs.trcmin * vs.maskT)

    if vs.enable_cyclic_x:
        for tracer in vs.npzd_tracers.values():
            cyclic.setcyclic_x(tracer)
