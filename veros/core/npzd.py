"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
import numpy as np  # NOTE np is already defined somehow
from .. import veros_method
from . import advection, diffusion, thermodynamics, cyclic, atmospherefluxes, utilities, isoneutral


@veros_method
def biogeochemistry(vs):
    """
    Integrate biochemistry: phytoplankton, zooplankton, detritus, po4
    """

    # Number of timesteps to do for bio tracers
    # nbio = int(vs.dt_mom // vs.dt_bio)
    nbio = int(vs.dt_tracer // vs.dt_bio)

    # Used to remineralize at the bottom - TODO calculate once elsewhere
    # Is there a better way to find the bottom?
    bottom_mask = np.zeros((vs.nx + 4, vs.ny + 4, vs.nz), dtype=np.bool)
    for k in range(vs.nz):
        bottom_mask[:, :, k] = (k == vs.kbot - 1)

    # temporary tracer object to store differences
    vs.temporary_tracers = {tracer: val[:, :, :, vs.tau].copy() for tracer, val in vs.npzd_tracers.items()}
    # Flags enable us to only work on tracers with a minimum available concentration
    flags = {tracer: np.ones_like(vs.temporary_tracers[tracer], dtype=np.bool) for tracer in vs.temporary_tracers}

    # Ensure positive data and keep flags of where
    for tracer, data in vs.temporary_tracers.items():
        flag_mask = (data > vs.trcmin) * vs.maskT
        flags[tracer][:, :, :] = flag_mask.astype(np.bool)
        data[:, :, :] = np.where(flag_mask, data, vs.trcmin)

    # Pre rules: Changes that need to be applied before running npzd dynamics
    pre_rules = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_pre_rules]
    for rule in pre_rules:
        for key, value in rule.items():
            vs.temporary_tracers[key][:, :, :] += value

    # How much plankton is blocking light
    plankton_total = sum([vs.temporary_tracers[plankton] for plankton in vs.plankton_types]) * vs.dzt

    # Integrated phytplankton - starting from top of layer going upwards
    # reverse cumulative sum because our top layer is the last.
    # Needs to be reversed again to reflect direction
    phyto_integrated = np.empty_like(vs.temporary_tracers["phytoplankton"])
    phyto_integrated[:, :, :-1] = np.cumsum(plankton_total[:, :, 1:][:, :, ::-1], axis=2)[:, :, ::-1]
    phyto_integrated[:, :, -1] = 0.0


    # TODO these could be created elsewhere
    # Dictionaries for storage of exported material
    export = {}
    bottom_export = {}
    impo = {}
    import_minus_export = {}

    # Temporary storage of mortality and recycled - to be used in rules
    # TODO make these local somehow, so they don't pollute the variable set
    vs.net_primary_production = {}
    vs.recycled = {}
    vs.mortality = {}

    # incomming shortwave radiation at top layer
    swr = vs.swr[:, :, np.newaxis] * np.exp(-vs.light_attenuation_phytoplankton * phyto_integrated)

    # light at top of grid box
    grid_light = swr * np.exp(vs.zw * vs.rctheta)

    # TODO is light_attenuation a bad name?
    # TODO we are doing almost the same cumsum as above, could we do it just once?
    light_attenuation = vs.dzt * vs.light_attenuation_water + vs.light_attenuation_phytoplankton * np.cumsum(plankton_total[:, :, ::-1], axis=2)[:, :, ::-1]

    # recycling rate determined according to b ** (cT)
    # zooplankton growth rate maxes out at 20C, TODO remove magic number
    bct = vs.bbio ** (vs.cbio * vs.temp[:, :, :, vs.taum1])
    bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, :, vs.taum1], 20))

    # Maximum grazing rate is a function of temperature
    # bctz sets an upper limit on effects of temperature on grazing
    gmax = vs.gbio * bctz

    jmax, avej = {}, {}

    for plankton, growth_function in vs.plankton_growth_functions.items():
        jmax[plankton], avej[plankton] = growth_function(vs, bct, grid_light, light_attenuation)


    # bio loop
    for _ in range(nbio):

        # Plankton is recycled, dying and growing
        for plankton in vs.plankton_types:

            # Nutrient limiting growth - if no limit, growth is determined by jmax
            u = np.inf

            # TODO Create "rule factory" to turn this into single rules used below
            for growth_limiting_function in vs.limiting_functions[plankton]:
                u = np.minimum(u, growth_limiting_function(vs, vs.temporary_tracers))

            vs.net_primary_production[plankton] = np.minimum(avej[plankton], u * jmax[plankton])

            # Fast recycling of plankton
            # TODO Could this be moved to individual rules?
            vs.recycled[plankton] = flags[plankton] * vs.nupt0 * bct * vs.temporary_tracers[plankton]  # TODO check nupt0 for other plankton types

            # Mortality of plankton
            # Would probably be easier to handle as rules
            vs.mortality[plankton] = flags[plankton] * vs.specific_mortality_phytoplankton * vs.temporary_tracers[plankton] # TODO proper mortality rates for other plankton


        # Detritus is recycled
        vs.recycled["detritus"] = flags["detritus"] * vs.nud0 * bct * vs.temporary_tracers["detritus"]

        # zooplankton displays quadric mortality rates
        vs.mortality["zooplankton"] = flags["zooplankton"] * vs.quadric_mortality_zooplankton * vs.temporary_tracers["zooplankton"] ** 2

        # TODO: move these to rules except grazing
        vs.grazing, vs.digestion, vs.excretion, vs.sloppy_feeding = \
                zooplankton_grazing(vs, vs.temporary_tracers, flags, gmax)
        vs.excretion_total = sum(vs.excretion.values())  # TODO this one should not be necessary with the rules


        # Fetch exported sinking material and calculate difference between layers
        # Amount of exported material is determined by cell z-height and sinking speed
        # amount falling through bottom is removed and remineralized later
        # impo is import from layer above. Only used to calculate difference
        for sinker, speed in vs.sinking_speeds.items():
            export[sinker] = speed * vs.dzt * vs.temporary_tracers[sinker] * flags[sinker]
            bottom_export[sinker] = export[sinker][bottom_mask]
            export[sinker][bottom_mask] -= bottom_export[sinker]

            impo[sinker] = np.empty_like(export[sinker])
            impo[sinker][:, :, -1] = 0
            impo[sinker][:, :, :-1] = export[sinker][:, :, :-1]
            import_minus_export[sinker] = impo[sinker] - export[sinker]

        # Gather all state updates
        npzd_updates = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_rules]

        # perform updates
        for update in npzd_updates:
            for key, value in update.items():
                vs.temporary_tracers[key][:, :, :] += value * vs.dt_bio

        # Import and export between layers
        for tracer in vs.sinking_speeds:
            vs.temporary_tracers[tracer][:, :, :] += import_minus_export[tracer]

        # Prepare temporary tracers for next bio iteration
        for tracer, data in vs.temporary_tracers.items():
            flag_mask = np.logical_and(flags[tracer], data > vs.trcmin) * vs.maskT
            flags[tracer] = flag_mask.astype(np.bool)  # np.where(flag_mask, True, False)
            data[:, :, :] = np.where(flag_mask, data, vs.trcmin)


    # Remineralize material fallen to the ocean floor
    vs.temporary_tracers["po4"][bottom_mask] += bottom_export["detritus"] * vs.redfield_ratio_PN
    if vs.enable_carbon:
        vs.temporary_tracers["DIC"][bottom_mask] += bottom_export["detritus"] * vs.redfield_ratio_CN
        vs.temporary_tracers["alkalinity"][bottom_mask] -= bottom_export["detritus"] * vs.redfield_ratio_CN

    # Post processesing or smoothing rules
    post_results = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_post_rules]
    for result in post_results:
        for key, value in result.items():
            vs.temporary_tracers[key][:, :, :] += value

    # Reset before returning
    for tracer, data in vs.temporary_tracers.items():
        flag_mask = np.logical_and(flags[tracer], data > vs.trcmin) * vs.maskT
        data[:, :, :] = np.where(flag_mask.astype(np.bool), data, vs.trcmin)

    # Return only the difference, as we may not override the results to be used in transport or elsewhere
    return {tracer: vs.temporary_tracers[tracer] - vs.npzd_tracers[tracer][:, :, :, vs.tau] for tracer in vs.npzd_tracers}

    # """ Get rid of k loop """



    # for k in reversed(range(vs.nz)):


    #     # Ensure positive data and keep flags of where
    #     for tracer, data in vs.temporary_tracers.items():
    #         flag_mask[tracer] = (data[:, :, k] > vs.trcmin) * vs.maskT[:, :, k]
    #         flags[tracer] = np.where(flag_mask[tracer], True, False)
    #         data[:, :, k] = np.where(flag_mask[tracer], data[:, :, k], vs.trcmin)

    #     # incomming radiation at layer
    #     swr[:, :] *= np.exp(- vs.light_attenuation_phytoplankton * phyto_integrated)


    #     # TODO remove if statements in running code
    #     if vs.enable_calcifiers:
    #         swr[:, :] *= np.exp(- vs.light_attenuation_caco3 * tracers["caco3"])
    #         dissk1 = vs.dissk0 * (1 - vs.Omega_c)
    #         # caco3_integrated[...] += np.maximum(tracers["caco3"], vs.trcmin) * vs.dzt[k]


    #     # Integration of all light blocking plankton in current layer. Values have been set positive above
    #     phyto_integrated[:, :] = sum([vs.temporary_tracers[plankton][:, :, k] for plankton in vs.plankton_types]) * vs.dzt[k]


    #     ## TODO: Update nud, bctz based on oxygen availability

    #     light at top of gridbox
    #     grid_light = swr * np.exp(vs.zw[k] * vs.rctheta)

    #     # calculate import of sinking particles from layer above
    #     for sinker in vs.sinking_speeds:
    #         impo[sinker] = export[sinker] / (vs.dzt[k] * vs.dt_bio)
    #         export[sinker][:, :] = 0


    #     # How much light gets through to this level
    #     light_attenuation = vs.dzt[k] * vs.light_attenuation_water + vs.light_attenuation_phytoplankton * phyto_integrated

    #     # if vs.enable_calcifiers:
    #     #     light_attenuation += vs.dzt[k] * vs.light_attenuation_caco3 * tracers["caco3"]

    #     # calculate maxing growth functions and averaged? growth
    #     jmax, avej = {}, {}
    #     for plankton, growth_function in vs.plankton_growth_functions.items():
    #         jmax[plankton], avej[plankton] = growth_function(vs, k, bct, grid_light, light_attenuation)


    #     # bio loop
    #     for _ in range(nbio):

    #         # Plankton is recycled, dying and growing
    #         for plankton in vs.plankton_types:

    #             # Nutrient limiting growth
    #             u = np.inf

    #             # TODO Create "rule factory" to turn this into single rules used below
    #             for growth_limiting_function in vs.limiting_functions[plankton]:
    #                 u = np.minimum(u, growth_limiting_function(vs, k, vs.temporary_tracers))

    #             vs.net_primary_production[plankton] = np.minimum(avej[plankton], u * jmax[plankton])

    #             # Fast recycling of plankton
    #             # TODO Could this be moved to individual rules?
    #             vs.recycled[plankton] = flags[plankton] * vs.nupt0 * bct[:, :, k] * vs.temporary_tracers[plankton][:, :, k]  # TODO check nupt0 for other plankton types

    #             # Mortality of plankton
    #             # Would probably be easier to handle as rules
    #             vs.mortality[plankton] = flags[plankton] * vs.specific_mortality_phytoplankton * vs.temporary_tracers[plankton][:, :, k]  # TODO proper mortality rates for other plankton

    #         # Detritus is recycled
    #         vs.recycled["detritus"] = flags["detritus"] * vs.nud0 * bct[:, :, k] * vs.temporary_tracers["detritus"][:, :, k]

    #         # zooplankton displays quadric mortality rates
    #         vs.mortality["zooplankton"] = flags["zooplankton"] * vs.quadric_mortality_zooplankton * vs.temporary_tracers["zooplankton"][:, :, k] ** 2

    #         # TODO: move these to rules except grazing
    #         vs.grazing, vs.digestion, vs.excretion, vs.sloppy_feeding = zooplankton_grazing(vs, k, vs.temporary_tracers, flags, gmax[:, :, k])
    #         vs.excretion_total = sum(vs.excretion.values())  # TODO this one should not be necessary with the rules

    #         # Temporary export for use in next layer
    #         tmp_expo = {sinker: speed[:, :, k] * vs.temporary_tracers[sinker][:, :, k] * flags[sinker] for sinker, speed in vs.sinking_speeds.items()}

    #         # Gather all state updates
    #         npzd_updates = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_rules]

    #         # perform updates
    #         for update in npzd_updates:
    #             for key, value in update.items():
    #                 # print(key, value[50,23] * vs.dt_bio)
    #                 vs.temporary_tracers[key][:, :, k] += value * vs.dt_bio

    #         # Import and export between layers
    #         for tracer in vs.sinking_speeds:
    #             vs.temporary_tracers[tracer][:, :, k] += impo[tracer] - tmp_expo[tracer]

    #             # Calculate total export from layer to next
    #             export[tracer][:, :] += tmp_expo[tracer] * vs.dt_bio

    #         # Prepare temporary tracers for next bio iteration
    #         for tracer, data in vs.temporary_tracers.items():
    #             flag_mask[tracer] = np.logical_and(flag_mask[tracer], data[:, :, k] > vs.trcmin) * vs.maskT[:, :, k]
    #             flags[tracer] = np.where(flag_mask[tracer], True, False)
    #             data[:, :, k] = np.where(flag_mask[tracer], data[:, :, k], vs.trcmin)


    #     # Remineralize at bottom, and ensure sinking stuff doesn't fall through
    #     # at_bottom = (vs.kbot - 1) == k
    #     at_bottom = (vs.kbot == k + 1) # TODO only calculate this once
    #     vs.temporary_tracers["po4"][at_bottom, k] += export["detritus"][at_bottom] * vs.redfield_ratio_PN

    #     if vs.enable_carbon:  # TODO remove ifs - maybe create common nutrient rule
    #         vs.temporary_tracers["DIC"][at_bottom, k] += export["detritus"][at_bottom] * vs.redfield_ratio_CN

    #     for sinker in export:
    #         export[sinker][:, :] -= at_bottom * export[sinker]

    #     # Calculate export for next layer
    #     # for sinker in export:
    #         export[sinker][:, :] *= vs.dzt[k] / nbio

    #     # if vs.enable_carbon:  # TODO remove ifs in process code
    #     #     # dprca = export["caco3"] * 1e-3
    #     #     dprca = vs.calpro / nbio #* 1e-3
    #     #     vs.prca += dprca * vs.dzt[k]
    #     #     vs.temporary_tracers["DIC"][:, :, k] -= dprca

    #     #     if not vs.enable_calcifiers:
    #     #         vs.temporary_tracers["alkalinity"][:, :, k] -= 2 * dprca

    #     #     vs.calpro[:, :] = 0

    # ## Remineralization of calcite - NOTE: Why can't this be in the main loop?
    # # if vs.enable_carbon:

    # #     vs.temporary_tracers["DIC"][:, :, :] += vs.prca[:, :, np.newaxis] * vs.rcak
    # #     vs.temporary_tracers["alkalinity"][:, :, :] += vs.prca[:, :, np.newaxis] * vs.rcak
    # #     vs.prca[:, :] = 0


    # post_results = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_post_rules]
    # for result in post_results:
    #     for key, value in result.items():
    #         vs.temporary_tracers[key][:, :, :] += value #* vs.dt_mom #/ vs.dzt[-1]


    # return {tracer: vs.temporary_tracers[tracer] - vs.npzd_tracers[tracer][:, :, :, vs.tau] for tracer in vs.npzd_tracers}

# TODO additional k-loops

# @veros_method
# def zooplankton_grazing_k(vs, k, tracers, flags, gmax):
#     zprefs = vs.zprefs

#     thetaZ = sum([pref_score * tracers[preference][:, :, k] for preference, pref_score in zprefs.items()]) + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

#     ingestion = {preference: pref_score / thetaZ for preference, pref_score in zprefs.items()}

#     grazing = {preference: flags[preference] * flags["zooplankton"] * gmax * ingestion[preference] * tracers[preference][:, :, k] * tracers["zooplankton"][:, :, k] for preference in ingestion}

#     digestion = {preference: vs.assimilation_efficiency * amount_grazed for preference, amount_grazed in grazing.items()}

#     excretion = {preference: vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

#     sloppy_feeding = {preference: (1 - vs.assimilation_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

#     return grazing, digestion, excretion, sloppy_feeding

@veros_method
def zooplankton_grazing(vs, tracers, flags, gmax):

    # TODO check saturation constants
    thetaZ = sum([pref_score * tracers[preference] for preference, pref_score in vs.zprefs.items()]) + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

    ingestion = {preference: pref_score / thetaZ for preference, pref_score in vs.zprefs.items()}

    grazing = {preference: flags[preference] * flags["zooplankton"] * gmax * ingestion[preference] * tracers[preference] * tracers["zooplankton"] for preference in ingestion}

    digestion = {preference: vs.assimilation_efficiency * amount_grazed for preference, amount_grazed in grazing.items()}

    excretion = {preference: vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

    sloppy_feeding = {preference: (1 - vs.assimilation_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

    return grazing, digestion, excretion, sloppy_feeding

# @veros_method
# def potential_growth(vs, k, bct, grid_light, light_attenuation, growth_parameter):
#     f1 = np.exp(-light_attenuation)
#     jmax = growth_parameter * bct[:, :, k]
#     gd = jmax * vs.dt_mom
#     avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

#     return jmax, avej

# @veros_method
# def phytoplankton_potential_growth(vs, k, bct, grid_light, light_attenuation):
#     return potential_growth(vs, k, bct, grid_light, light_attenuation, vs.abio_P)

# @veros_method
# def coccolitophore_potential_growth(vs, k, bct, grid_light, light_attenuation):
#     return potential_growth(vs, k, bct, grid_light, light_attenuation, vs.abio_C)


# @veros_method
# def diazotroph_potential_growth(vs, k, bct, grid_light, light_attenuation):
#     f1 = np.exp(-light_attenuation)
#     jmax = np.maximum(0, vs.abio_P * vs.jdiar * (bct[:, :, k] - 2.6))
#     gd = np.maximum(1e-14, jmax * vs.dt_mom)
#     avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

#     return jmax, avej

@veros_method
def potential_growth(vs, bct, grid_light, light_attenuation, growth_parameter):
    f1 = np.exp(-light_attenuation)
    jmax = growth_parameter * bct
    #gd = jmax * vs.dt_mom
    gd = jmax * vs.dt_tracer
    avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

    return jmax, avej

@veros_method
def phytoplankton_potential_growth(vs, bct, grid_light, light_attenuation):
    return potential_growth(vs, bct, grid_light, light_attenuation, vs.abio_P)

@veros_method
def coccolitophore_potential_growth(vs, bct, grid_light, light_attenuation):
    return potential_growth(vs, bct, grid_light, light_attenuation, vs.abio_C)


@veros_method
def diazotroph_potential_growth(vs, k, bct, grid_light, light_attenuation):
    f1 = np.exp(-light_attenuation)
    jmax = np.maximum(0, vs.abio_P * vs.jdiar * (bct - 2.6))
    # gd = np.maximum(1e-14, jmax * vs.dt_mom)
    gd = np.maximum(1e-14, jmax * vs.dt_tracer)
    avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

    return jmax, avej

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

# @veros_method
# def phosphate_limitation_phytoplankton(vs, k, tracers):
#     return general_nutrient_limitation(tracers["po4"][:, :, k], vs.saturation_constant_N / vs.redfield_ratio_PN)

# @veros_method
# def phosphate_limitation_coccolitophore(vs, k, tracers):
#     return general_nutrient_limitation(tracers["po4"][:, :, k], vs.saturation_constant_NC / vs.redfield_ratio_PN)

# @veros_method
# def phosphate_limitation_diazotroph(vs, k, tracers):
#     return general_nutrient_limitation(tracers["po4"][:, :, k], vs.saturation_constant_N / vs.redfield_ratio_PN)

# @veros_method
# def nitrate_limitation_diazotroph(vs, k, tracers):
#     return general_nutrient_limitation(tracers["no3"][:, :, k], vs.saturation_constant_N)

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

# @veros_method
# def update_flags_and_tracers(vs, flags, tracers, k, refresh=False):
#     """Set flags"""

#     for key in tracers:
#         keep = flags[key] if not refresh else True  # if the flag was already false, keep it false
#         flags[key] = ((tracers[key] * vs.maskT[:, :, k]) > vs.trcmin) * keep
#         tracers[key] = np.maximum(tracers[key] * vs.maskT[:, :, k], vs.trcmin)

@veros_method
def register_npzd_data(vs, name, value=None):

    if name not in vs.npzd_tracers:
        if vs.show_npzd_graph:
            vs.npzd_graph.attr('node', shape='square')
            vs.npzd_graph.node(name)
    else:
        print(name, "has already been added to the NPZD data set, value has been updated")

    if value is None:
        value = np.zeros_like(vs.phytoplankton)

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

# @veros_method
# def calpro(vs, plankton, calcite):
#     # return {cal: (vs.mortality[plankton] + vs.sloppy_feeding[plankton]) * vs.capr * vs.redfield_ratio_CN * 1e3}

#     vs.calpro += (vs.mortality[plankton] + vs.grazing[plankton] * (1 - vs.assimilation_efficiency)) * vs.capr * vs.redfield_ratio_CN #* 1.e3
#     # return {cal: vs.calpro}

@veros_method
def calcite_production(vs, plankton, DIC, calcite):
    """ Calcite is produced at a rate similar to detritus"""
    # calprod = (vs.sloppy_feeding[plankton] + vs.mortality[plankton] + vs.sloppy_feeding["zooplankton"] + vs.mortality["zooplankton"]) * vs.capr * vs.redfield_ratio_CN
    #nbio = int(vs.dt_mom // vs.dt_bio)
    #dprca = calprod / nbio

    dprca = (vs.mortality[plankton] + vs.grazing[plankton] * (1 - vs.assimilation_efficiency))  * vs.capr * vs.redfield_ratio_CN #/ vs.dt_mom

    return {DIC: -dprca, calcite: dprca}

@veros_method
def calcite_production_alk(vs, plankton, alkalinity, calcite):
    return {alkalinity: 2 * calcite_production(vs, plankton, "DIC", calcite)["DIC"]}

@veros_method
def calcite_production_phyto(vs, DIC, calcite):
    return calcite_production(vs, "phytoplankton", DIC, calcite)

@veros_method
def calcite_production_phyto_alk(vs, alkalinity, calcite):
    return calcite_production_alk(vs, "phytoplankton", alkalinity, calcite)

@veros_method
def post_redistribute_calcite(vs, calcite, tracer):
    total_production = (vs.temporary_tracers[calcite] * vs.dzt).sum(axis=2)
    redistributed_production = total_production[:, :, np.newaxis] * vs.rcak
    return {tracer: redistributed_production}

@veros_method
def pre_reset_calcite(vs, tracer, calcite):
    return {calcite: - vs.temporary_tracers[calcite]}

@veros_method
def recycling_to_alk(vs, detritus, alkalinity):
    """ This should be turned into a combined rule with DIC, as they just have opposite terms, but only withdraw from detritus once """
    return {alkalinity: - recycling_to_dic(vs, detritus, "DIC")["DIC"]}

@veros_method
def primary_production_from_alk(vs, alkalinity, plankton):
    """ Single entry, should be merged with DIC """
    return {alkalinity: - primary_production_from_DIC(vs, "DIC", plankton)["DIC"]}

@veros_method
def recycling_phyto_to_alk(vs, plankton, alkalinity):
    """ Single entry should be merged with DIC """
    return {alkalinity: - recycling_phyto_to_dic(vs, plankton, "DIC")["DIC"]}

@veros_method
def excretion_alk(vs, plankton, alkalinity):
    """ Single entry, should be merged with DIC """
    return {alkalinity: -excretion_dic(vs, plankton, "DIC")["DIC"]}

@veros_method
def co2_surface_flux(vs, co2, dic):
    # TODO not global please
    vs.cflux[:, :] = atmospherefluxes.carbon_flux(vs)
    flux = np.zeros((vs.cflux.shape[0], vs.cflux.shape[1], vs.nz))
    # flux[:, :, -1] = vs.cflux * vs.dt_mom / vs.dzt[-1]
    flux[:, :, -1] = vs.cflux * vs.dt_tracer / vs.dzt[-1]
    return {dic: flux}

@veros_method
def co2_surface_flux_alk(vs, co2, alk):
    flux = np.zeros((vs.cflux.shape[0], vs.cflux.shape[1], vs.nz))
    flux[:, :, -1] = - vs.cflux * vs.dt_tracer / vs.dzt[-1]
    return {alk: flux}

@veros_method
def setupNPZD(vs):
    """Taking veros variables and packaging them up into iterables"""
    vs.npzd_tracers = {}  # Dictionary keeping track of plankton, nutrients etc.
    vs.npzd_rules = []  # List of rules describing the interaction between tracers
    vs.npzd_pre_rules = []
    vs.npzd_post_rules = []

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
        # vs.prca = np.zeros_like(vs.dic[:, :, 0])
        #vs.prca = np.zeros_like(vs.dic[:, :, 0, 0])
        dcaco3 = 6500.0 # remineralization depth of calcite [m]

        # vs.rcak = np.empty_like(vs.dic)
        vs.rcak = np.empty_like(vs.dic[...,0])
        zw = vs.zw - vs.dzt  # using dzt because dzw is weird
        vs.rcak[:, :, :-1] = (- np.exp(zw[:-1] / dcaco3) + np.exp(zw[1:] / dcaco3)) / vs.dzt[:-1]
        vs.rcak[:, :, -1] = - (np.exp(zw[-1] / dcaco3) - 1.0) / vs.dzt[-1]

        rcab = np.empty_like(vs.dic[..., 0])
        rcab[:, : -1] = 1 / vs.dzt[-1]
        rcab[:, :, :-1] = np.exp(zw[:-1] / dcaco3) / vs.dzt[:-1]

        vs.rcak[vs.kbot - 1] = rcab[vs.kbot - 1]
        vs.rcak[...] *= vs.maskT


        # Need to track dissolved inorganic carbon, alkalinity and calcium carbonate
        register_npzd_data(vs, "DIC", vs.dic)
        register_npzd_data(vs, "alkalinity", vs.alkalinity)



        vs.npzd_pre_rules.append((co2_surface_flux, "co2", "DIC"))
        vs.npzd_pre_rules.append((co2_surface_flux_alk, "co2", "alkalinity"))

        register_npzd_rule(vs, recycling_to_dic, "detritus", "DIC", label="Remineralization")
        register_npzd_rule(vs, primary_production_from_DIC, "DIC", "phytoplankton", label="Primary production")
        register_npzd_rule(vs, recycling_phyto_to_dic, "phytoplankton", "DIC", label="Fast recycling")
        register_npzd_rule(vs, excretion_dic, "zooplankton", "DIC", label="Excretion")


        register_npzd_rule(vs, recycling_to_alk, "detritus", "alkalinity", label="Remineralization")
        register_npzd_rule(vs, primary_production_from_alk, "alkalinity", "phytoplankton", label="Primary production")
        register_npzd_rule(vs, recycling_phyto_to_alk, "phytoplankton", "alkalinity", label="Fast recycling")
        register_npzd_rule(vs, excretion_alk, "zooplankton", "alkalinity", label="Excretion")


        # These rules will be different if we track coccolithophores
        if not vs.enable_calcifiers:
            register_npzd_data(vs, "caco3", vs.caco3)

            register_npzd_rule(vs, calcite_production_phyto, "DIC", "caco3", label="Production of calcite")
            register_npzd_rule(vs, calcite_production_phyto_alk, "alkalinity", "caco3", label="Production of calcite")
            vs.npzd_post_rules.append((post_redistribute_calcite, "caco3", "alkalinity"))
            vs.npzd_post_rules.append((post_redistribute_calcite, "caco3", "DIC"))
            vs.npzd_pre_rules.append((pre_reset_calcite, "caco3", "caco3"))



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

    # Keep derivatives of everything..
    vs.npzd_tracer_derivatives = {tracer: np.zeros_like(data) for tracer, data in vs.npzd_tracers.items()}

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

    # TODO: Refactor transportation code to be defined only once and also used by thermodynamics


    npzd_changes = biogeochemistry(vs)

    if vs.enable_neutral_diffusion:
        isoneutral.isoneutral_diffusion_pre(vs)  # TODO remove this outside the loop or ensure, it is called by thermodynamics first



    # TODO: move to function. This is essentially the same as vmix in thermodynamics

    """
    For vertical mixing
    """
    a_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
    b_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
    c_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
    d_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
    delta = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)

    ks = vs.kbot[2:-2, 2:-2] - 1
    delta[:, :, :-1] = vs.dt_tracer / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.kappaH[2:-2, 2:-2, :-1]
    delta[:, :, -1] = 0
    a_tri[:, :, 1:] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:]
    b_tri[:, :, 1:] = 1 + (delta[:, :, 1:] + delta[:, :, :-1]) / vs.dzt[np.newaxis, np.newaxis, 1:]
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1]

    for tracer, tracer_data in vs.npzd_tracers.items():

        """
        Advection of tracers
        """
        # TODO: Rename npzd_tracer_derivatives to npzd_advection_derivates? or something like that
        thermodynamics.advect_tracer(vs, tracer_data[:, :, :, vs.tau], vs.npzd_tracer_derivatives[tracer][:, :, :, vs.tau])

        # TODO vs.dt_tracer ??
        # Adam-Bashforth timestepping
        # tracer_data[:, :, :, vs.taup1] = tracer_data[:, :, :, vs.tau] + vs.dt_mom \
        tracer_data[:, :, :, vs.taup1] = tracer_data[:, :, :, vs.tau] + vs.dt_tracer \
                * ((1.5 + vs.AB_eps) * vs.npzd_tracer_derivatives[tracer][:, :, :, vs.tau]
                - (0.5 + vs.AB_eps) * vs.npzd_tracer_derivatives[tracer][:, :, :, vs.taum1]) * vs.maskT

        """
        Diffusion of tracers
        """

        # TODO distinguish between biharmonic mixing and simple diffusion like in thermodynamics
        diffusion_change = np.empty_like(tracer_data[:, :, :, 0])
        diffusion.biharmonic(vs, tracer_data[:, :, :, vs.tau], np.sqrt(abs(vs.K_hbi)), diffusion_change)

        # tracer_data[:, :, :, vs.taup1] += vs.dt_mom * diffusion_change
        tracer_data[:, :, :, vs.taup1] += vs.dt_tracer * diffusion_change

        """
        Isopycnal diffusion
        """

        if vs.enable_neutral_diffusion:
            dtracer_iso = np.zeros_like(tracer_data[..., 0])

            # NOTE isoneutral_diffusion_decoupled is a temporary solution to splitting the explicit dependence on time and salinity from the function isoneutral_diffusion
            isoneutral.isoneutral_diffusion_decoupled(vs, tracer_data, dtracer_iso, iso=True, skew=False)

            if vs.enable_skew_diffusion:
                dtracer_skew = np.zeros_like(tracer_data[..., 0])
                isoneutral.isoneutral_diffusion_decoupled(vs, tracer_data, dtracer_skew, iso=False, skew=True)





        """
        Vertical mixing of tracers
        """
        d_tri[:, :, :] = tracer_data[2:-2, 2:-2, :, vs.taup1]
        # TODO: surface flux?
        # d_tri[:, :, -1] += surface_forcing
        sol, mask = utilities.solve_implicit(vs, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)

        tracer_data[2:-2, 2:-2, :, vs.taup1] = utilities.where(vs, mask, sol, tracer_data[2:-2, 2:-2, :, vs.taup1])

    for tracer, change in npzd_changes.items():
        vs.npzd_tracers[tracer][:, :, :, vs.taup1] += change


    for tracer in vs.npzd_tracers.values():
        tracer[:, :, :, vs.taup1] = np.maximum(tracer[:, :, :, vs.taup1], vs.trcmin * vs.maskT)

    if vs.enable_cyclic_x:
        for tracer in vs.npzd_tracers.values():
            cyclic.setcyclic_x(tracer)
