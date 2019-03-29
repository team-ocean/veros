"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
import numpy as np  # NOTE np is already defined somehow
from .. import veros_method
from .. import time
from . import diffusion, thermodynamics, cyclic, utilities, isoneutral


@veros_method
def biogeochemistry(vs):
    """
    Integrate biochemistry: phytoplankton, zooplankton, detritus, po4
    """

    # declination and fraction of day with daylight
    declin = np.sin((np.mod(vs.time * time.SECONDS_TO_X["years"], 1) - 0.22) * 2.0 * np.pi) * 0.4
    rctheta = np.maximum(-1.5, np.minimum(1.5, np.radians(vs.yt) - declin))

    # 1.33 is derived from Snells law for the air-sea barrier
    vs.rctheta = vs.light_attenuation_water / np.sqrt(1.0 - (1.0 - np.cos(rctheta)**2.0) / 1.33**2)
    dayfrac = np.minimum(1.0, -np.tan(np.radians(vs.yt)) * np.tan(declin))
    vs.dayfrac = np.maximum(1e-12, np.arccos(np.maximum(-1.0, dayfrac)) / np.pi)

    # Number of timesteps to do for bio tracers
    nbio = int(vs.dt_tracer // vs.dt_bio)

    # Used to remineralize at the bottom - TODO calculate once elsewhere
    bottom_mask = np.empty((vs.nx + 4, vs.ny + 4, vs.nz), dtype=np.bool)
    bottom_mask[:, :, :] = np.arange(vs.nz)[np.newaxis, np.newaxis, :] == (vs.kbot - 1)[:, :, np.newaxis]

    # temporary tracer object to store differences
    vs.temporary_tracers = {tracer: val[:, :, :, vs.tau].copy()
                            for tracer, val in vs.npzd_tracers.items()}

    # Flags enable us to only work on tracers with a minimum available concentration
    flags = {tracer: np.ones_like(vs.temporary_tracers[tracer], dtype=np.bool)
             for tracer in vs.temporary_tracers}

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
    phyto_integrated[:, :, :-1] = plankton_total[:, :, 1:]
    phyto_integrated[:, :, -1] = 0.0

    # TODO these could be created elsewhere
    # Dictionaries for storage of exported material
    export = {}
    bottom_export = {}
    impo = {}
    import_minus_export = {}


    # incomming shortwave radiation at top layer
    swr = vs.swr[:, :, np.newaxis] * np.exp(-vs.light_attenuation_phytoplankton
                                            * np.cumsum(phyto_integrated[:, :, ::-1], axis=2)[:, :, ::-1])

    # light at top of grid box
    grid_light = swr * np.exp(vs.zw[np.newaxis, np.newaxis, :]
                              * vs.rctheta[np.newaxis, :, np.newaxis])

    # TODO is light_attenuation a bad name?
    light_attenuation = vs.dzt * vs.light_attenuation_water + plankton_total

    # recycling rate determined according to b ** (cT)
    bct = vs.bbio ** (vs.cbio * vs.temp[:, :, :, vs.tau])
    bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, :, vs.tau],
                                            vs.zooplankton_max_growth_temp))

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

            # Nutrient limiting growth - if no limit, growth is determined by avej
            u = 1

            # TODO Could this somehow be turned into a single rule? I think it would be possible
            for growth_limiting_function in vs.limiting_functions[plankton]:
                u = np.minimum(u, growth_limiting_function(vs, vs.temporary_tracers))

            vs.net_primary_production[plankton] = flags[plankton] * flags["po4"]\
                * np.minimum(avej[plankton], u * jmax[plankton]) * vs.temporary_tracers[plankton]

            # Fast recycling of plankton
            # TODO Could this be moved to individual rules?
            vs.recycled[plankton] = flags[plankton] * vs.recycling_rates[plankton] * bct\
                * vs.temporary_tracers[plankton]

            # Mortality of plankton
            vs.mortality[plankton] = flags[plankton] * vs.mortality_rates[plankton]\
                * vs.temporary_tracers[plankton]


        # Detritus is recycled
        vs.recycled["detritus"] = flags["detritus"] * vs.recycling_rates["detritus"] * bct\
            * vs.temporary_tracers["detritus"]

        # zooplankton displays quadric mortality rates
        vs.mortality["zooplankton"] = flags["zooplankton"] * vs.quadric_mortality_zooplankton\
            * vs.temporary_tracers["zooplankton"] ** 2

        # TODO: move these to rules except grazing
        vs.grazing, vs.digestion, vs.excretion, vs.sloppy_feeding = \
            zooplankton_grazing(vs, vs.temporary_tracers, flags, gmax)
        vs.excretion_total = sum(vs.excretion.values())

        # Fetch exported sinking material and calculate difference between layers
        # Amount of exported material is determined by cell z-height and sinking speed
        # amount falling through bottom is removed and remineralized later
        # impo is import from layer above. Only used to calculate difference
        for sinker, speed in vs.sinking_speeds.items():
            export[sinker] = speed * vs.temporary_tracers[sinker] * flags[sinker] / vs.dzt
            bottom_export[sinker] = export[sinker] * bottom_mask
            export[sinker][...] -= bottom_export[sinker]
            # vs.temporary_tracers[sinker][bottom_mask] = 0
            # vs.temporary_tracers[sinker][...] -= bottom_export[sinker]

            impo[sinker] = np.empty_like(export[sinker])
            impo[sinker][:, :, -1] = 0
            impo[sinker][:, :, :-1] = export[sinker][:, :, 1:] * (vs.dzt[1:] / vs.dzt[:-1])

            import_minus_export[sinker] = impo[sinker] - export[sinker]

        # Gather all state updates
        npzd_updates = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_rules]

        # perform updates
        for update in npzd_updates:
            for key, value in update.items():
                vs.temporary_tracers[key][:, :, :] += value * vs.dt_bio

        # Import and export between layers
        for tracer in vs.sinking_speeds:
            vs.temporary_tracers[tracer][:, :, :] += import_minus_export[tracer] * vs.dt_bio

        # Prepare temporary tracers for next bio iteration
        for tracer, data in vs.temporary_tracers.items():
            flag_mask = np.logical_and(flags[tracer], data > vs.trcmin) * vs.maskT
            flags[tracer] = flag_mask.astype(np.bool)
            data[:, :, :] = np.where(flag_mask, data, vs.trcmin)

        # Remineralize material fallen to the ocean floor
        vs.temporary_tracers["po4"][...] += bottom_export["detritus"] * vs.redfield_ratio_PN * vs.dt_bio
        if vs.enable_carbon:
            vs.temporary_tracers["DIC"][...] += bottom_export["detritus"] * vs.redfield_ratio_CN * vs.dt_bio

    # Post processesing or smoothing rules
    post_results = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_post_rules]
    for result in post_results:
        for key, value in result.items():
            vs.temporary_tracers[key][:, :, :] += value

    # Reset before returning
    for tracer, data in vs.temporary_tracers.items():
        flag_mask = np.logical_and(flags[tracer], data > vs.trcmin) * vs.maskT
        data[:, :, :] = np.where(flag_mask.astype(np.bool), data, vs.trcmin)

    """
    Only return the difference. Will be added to timestep taup1
    """
    return {tracer: vs.temporary_tracers[tracer] - vs.npzd_tracers[tracer][:, :, :, vs.tau]
            for tracer in vs.npzd_tracers}


@veros_method
def zooplankton_grazing(vs, tracers, flags, gmax):
    """
    Zooplankton grazing returns total grazing, digestion i.e. how much is available
    for zooplankton growth, excretion and sloppy feeding
    All are useful to have calculated once and made available to rules
    """

    # TODO check saturation constants
    thetaZ = sum([pref_score * tracers[preference] for preference, pref_score in vs.zprefs.items()])\
        + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

    ingestion = {preference: pref_score / thetaZ for preference, pref_score in vs.zprefs.items()}

    grazing = {preference: flags[preference] * flags["zooplankton"] * gmax *
               ingestion[preference] * tracers[preference] * tracers["zooplankton"]
               for preference in ingestion}

    digestion = {preference: vs.assimilation_efficiency * amount_grazed
                 for preference, amount_grazed in grazing.items()}

    # excretion = {preference: vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency)
    #              * amount_grazed for preference, amount_grazed in grazing.items()}
    excretion = {preference: (1 - vs.assimilation_efficiency) * amount_digested
                 for preference, amount_digested in digestion.items()}

    # sloppy_feeding = {preference: (1 - vs.assimilation_efficiency) * amount_grazed
    #                   for preference, amount_grazed in grazing.items()}
    sloppy_feeding = {preference: grazing[preference] - digestion[preference] for preference in grazing}

    return grazing, digestion, excretion, sloppy_feeding


@veros_method
def potential_growth(vs, bct, grid_light, light_attenuation, growth_parameter):
    """ Potential growth of phytoplankton """
    f1 = np.exp(-light_attenuation)
    jmax = growth_parameter * bct
    gd = jmax * vs.dayfrac[np.newaxis, :, np.newaxis]  # growth in fraction of day
    avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

    return jmax, avej


@veros_method
def phytoplankton_potential_growth(vs, bct, grid_light, light_attenuation):
    """ Regular potential growth scaled by vs.abi_P """
    return potential_growth(vs, bct, grid_light, light_attenuation, vs.abio_P)


@veros_method
def coccolitophore_potential_growth(vs, bct, grid_light, light_attenuation):
    """ Scale potential growth by vs.abio_C """
    return potential_growth(vs, bct, grid_light, light_attenuation, vs.abio_C)


@veros_method
def diazotroph_potential_growth(vs, bct, grid_light, light_attenuation):
    """ Potential growth of diazotroph is limited by a minimum temperature """
    f1 = np.exp(-light_attenuation)
    jmax = np.maximum(0, vs.abio_P * vs.jdiar * (bct - vs.bct_min_diaz))
    gd = np.maximum(vs.gd_min_diaz, jmax)
    avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

    return jmax, avej


@veros_method
def avg_J(vs, f1, gd, grid_light, light_attenuation):
    """Average J"""
    u1 = np.maximum(grid_light / gd, vs.u1_min)
    u2 = u1 * f1

    # NOTE: There is an approximation here: u1 < 20
    phi1 = np.log(u1 + np.sqrt(1 + u1**2)) - (np.sqrt(1 + u1**2) - 1) / u1
    phi2 = np.log(u2 + np.sqrt(1 + u2**2)) - (np.sqrt(1 + u2**2) - 1) / u2

    return gd * (phi1 - phi2) / light_attenuation


def general_nutrient_limitation(nutrient, saturation_constant):
    """ Nutrient limitation form for all nutrients """
    return nutrient / (saturation_constant + nutrient)

# TODO replace vs.saturation_constant with something, that makes sense
@veros_method
def phosphate_limitation_phytoplankton(vs, tracers):
    """ Phytoplankton limit to growth by phosphate limitation """
    return general_nutrient_limitation(tracers["po4"], vs.saturation_constant_N * vs.redfield_ratio_PN)


@veros_method
def phosphate_limitation_coccolitophore(vs, tracers):
    """ Coccolitophore limit to growth by phosphate limitation """
    return general_nutrient_limitation(tracers["po4"], vs.saturation_constant_NC * vs.redfield_ratio_PN)


@veros_method
def phosphate_limitation_diazotroph(vs, tracers):
    """ Diazotroph limit to growth by phosphate limitation """
    return general_nutrient_limitation(tracers["po4"], vs.saturation_constant_N * vs.redfield_ratio_PN)


@veros_method
def nitrate_limitation_diazotroph(vs, tracers):
    """ Diazotroph limit to growth by nitrate limitation """
    return general_nutrient_limitation(tracers["no3"], vs.saturation_constant_N)


@veros_method
def dop_limitation_phytoplankton(vs, tracers):
    """ Phytoplankton limit to growth by DOP limitation """
    return vs.hdop * general_nutrient_limitation("DOP", vs.saturation_constant_N / vs.redfield_ratio_PN)


@veros_method
def dop_limitation_coccolitophore(vs, tracers):
    """ Phytoplankton limit to growth by DOP limitation """
    return vs.hdop * general_nutrient_limitation("DOP", vs.saturation_constant_NC / vs.redfield_ratio_PN)


@veros_method
def maximized_dop_po4_limitation_phytoplankton(vs, tracers):
    """ Consumption of nutrient switches, by which is the largest - remember to handle in rules """
    lim_po4 = phosphate_limitation_phytoplankton(vs, tracers)
    lim_DOP = dop_limitation_phytoplankton(vs, tracers)
    vs.dop_consumption = lim_DOP > lim_po4
    return np.where(vs.dop_consumption, lim_DOP, lim_po4)


@veros_method
def maximized_dop_po4_limitation_coccolitophre(vs, tracers):
    """ Like maximized_dop_po4_limitation_phytoplankton but for coccolitophores """
    return np.maximum(phosphate_limitation_coccolitophore(vs, tracers),
                      dop_limitation_coccolitophore(vs, tracers))


@veros_method
def register_npzd_data(vs, name, value=None, cluster=None):
    """
    Add tracer to the NPZD data set and create node in interaction graph
    Tracers added are available in the npzd dynamics and is automatically
    included in transport equations
    """

    if name not in vs.npzd_tracers:
        if vs.show_npzd_graph:
            with vs.npzd_graph.subgraph(name="cluster_"+cluster if cluster else None) as graph:
                graph.attr('node', shape='square')
                graph.node(name)
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
        vs.npzd_graph.edge(source, destination, xlabel=label, lblstyle="above, sloped")
        # it is also possible to add tooltiplabels for more explanation

@veros_method
def register_npzd_post_rule(vs, function, source, destination, label="?"):
    vs.npzd_post_rules.append((function, source, destination))
    if vs.show_npzd_graph:
        vs.npzd_graph.edge(source, destination, style='dotted', xlabel=label)


@veros_method
def register_npzd_pre_rule(vs, function, source, destination, label="?"):
    vs.npzd_pre_rules.append((function, source, destination))
    if vs.show_npzd_graph:
        vs.npzd_graph.edge(source, destination, style='dotted', xlabel=label)


@veros_method
def setup_basic_npzd_rules(vs):
    """
    Setup rules for basic NPZD model including phosphate, detritus, phytoplankton and zooplankton
    """
    from .npzd_rules import grazing, mortality, sloppy_feeding, recycling_to_po4, \
        zooplankton_self_grazing, excretion, primary_production

    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
    vs.sinking_speeds["detritus"] = (vs.wd0 + vs.mw * np.where(-zw < vs.mwz, -zw, vs.mwz)) \
        # * vs.maskT

    # Add "regular" phytoplankton to the model
    vs.plankton_types = ["phytoplankton"]  # Phytoplankton types in the model. For blocking light
    vs.plankton_growth_functions["phytoplankton"] = phytoplankton_potential_growth
    vs.limiting_functions["phytoplankton"] = [phosphate_limitation_phytoplankton]
    vs.recycling_rates["phytoplankton"] = vs.nupt0
    vs.recycling_rates["detritus"] = vs.nud0
    vs.mortality_rates["phytoplankton"] = vs.specific_mortality_phytoplankton

    # Zooplankton preferences for grazing on keys
    # Values are scaled automatically at the end of this function
    vs.zprefs = {"phytoplankton": vs.zprefP, "zooplankton": vs.zprefZ, "detritus": vs.zprefDet}

    # Register for basic model
    register_npzd_data(vs, "detritus", vs.detritus)
    register_npzd_data(vs, "phytoplankton", vs.phytoplankton, cluster="plankton")
    register_npzd_data(vs, "zooplankton", vs.zooplankton, cluster="plankton")
    register_npzd_data(vs, "po4", vs.po4, cluster="nutrients")

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

    if not vs.enable_nitrogen:
        register_npzd_rule(vs, primary_production, "po4", "phytoplankton", label="Primary production")


@veros_method
def setup_carbon_npzd_rules(vs):
    """
    Rules for including a carbon cycle
    """
    # TODO should alkalinity rules be moved to a post rule?
    # The actual action is on DIC, but the to variables overlap
    from .npzd_rules import co2_surface_flux, co2_surface_flux_alk, recycling_to_dic, \
        primary_production_from_DIC, excretion_dic, recycling_phyto_to_dic, \
        primary_production_from_alk, recycling_to_alk, recycling_phyto_to_alk, excretion_alk, \
        dic_alk_scale

    # vs.dcaco3 = 3500 
    # TODO create as variable
    vs.rcak = np.empty_like(vs.dic[..., 0])
    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
    vs.rcak[:, :, :-1] = (- np.exp(zw[:-1] / vs.dcaco3) + np.exp(zw[1:] / vs.dcaco3)) / vs.dzt[:-1]
    vs.rcak[:, :, -1] = - (np.exp(zw[-1] / vs.dcaco3) - 1.0) / vs.dzt[-1]

    rcab = np.empty_like(vs.dic[..., 0])
    rcab[:, : -1] = 1 / vs.dzt[-1]
    rcab[:, :, :-1] = np.exp(zw[:-1] / vs.dcaco3) / vs.dzt[1:]

    bottom_mask = np.empty((vs.nx + 4, vs.ny + 4, vs.nz), dtype=np.bool)
    bottom_mask[:, :, :] = np.arange(vs.nz)[np.newaxis, np.newaxis, :] == (vs.kbot - 1)[:, :, np.newaxis]

    vs.rcak[bottom_mask] = rcab[bottom_mask]
    vs.rcak[...] *= vs.maskT

    # Need to track dissolved inorganic carbon, alkalinity
    register_npzd_data(vs, "DIC", vs.dic, cluster="nutrients")
    register_npzd_data(vs, "alkalinity", vs.alkalinity)

    # Exchange of CO2 with the atmosphere
    vs.npzd_pre_rules.append((co2_surface_flux, "co2", "DIC"))

    # Common rule set for nutrient
    register_npzd_rule(vs, recycling_to_dic, "detritus", "DIC", label="Remineralization")
    register_npzd_rule(vs, primary_production_from_DIC, "DIC", "phytoplankton", label="Primary production")
    register_npzd_rule(vs, recycling_phyto_to_dic, "phytoplankton", "DIC", label="Fast recycling")
    register_npzd_rule(vs, excretion_dic, "zooplankton", "DIC", label="Excretion")

    register_npzd_post_rule(vs, dic_alk_scale, "DIC", "alkalinity")
    # These rules will be different if we track coccolithophores
    if not vs.enable_calcifiers:
        from .npzd_rules import calcite_production_phyto, calcite_production_phyto_alk, \
                post_redistribute_calcite, pre_reset_calcite

        register_npzd_data(vs, "caco3", np.zeros_like(vs.dic))  # Only for collection purposes

        # Collect calcite produced by phytoplankton and zooplankton and redistribute it
        register_npzd_rule(vs, calcite_production_phyto, "DIC", "caco3", label="Production of calcite")
        register_npzd_rule(vs, calcite_production_phyto_alk, "alkalinity", "caco3", label="Production of calcite")
        # TODO create register_post_rule and register_pre_rule
        register_npzd_post_rule(vs, post_redistribute_calcite, "caco3", "alkalinity", label="dissolution")
        register_npzd_post_rule(vs, post_redistribute_calcite, "caco3", "DIC", label="dissolution")
        register_npzd_pre_rule(vs, pre_reset_calcite, "caco3", "caco3", "reset")


@veros_method
def setup_nitrogen_npzd_rules(vs):
    """ Rules for including diazotroph, nitrate """
    from .npzd_rules import recycling_to_no3, empty_rule, grazing, recycling_to_po4, excretion, \
        mortality, primary_production_from_dop_po4, primary_production_from_po4_dop
    # TODO complete rules:
    #       - Primary production rules need to be generalized
    #       - DOP, DON availability needs to be considered

    register_npzd_data(vs, "diazotroph", vs.diazotroph)
    register_npzd_data(vs, "no3", vs.no3)
    register_npzd_data(vs, "DOP", vs.dop)
    register_npzd_data(vs, "DON", vs.don)

    vs.mortality_rates["diazotroph"] = vs.specific_mortality_diazotroph
    vs.recycling_rates["diazotroph"] = vs.nupt0_D
    vs.recycling_rates["DOP"] = vs.nudop0
    vs.recycling_rates["DON"] = vs.nudon0

    vs.zprefs["diazotroph"] = vs.zprefD  # Add preference for zooplankton to graze on diazotrophs
    vs.plankton_types.append("diazotroph")  # Diazotroph behaces like plankton
    vs.plankton_growth_functions["diazotroph"] = phytoplankton_potential_growth  # growth function

    # Limited in nutrients by both phosphate and nitrate
    vs.limiting_functions["diazotroph"] = [maximized_dop_po4_limitation_phytoplankton,
                                           nitrate_limitation_diazotroph]

    # Replace po4 limitation for phytoplankton with a combined limitation with DOP
    # TODO do the same for coccolitophore
    vs.limiting_functions["phytoplankton"] = list(filter(lambda lim: lim != phosphate_limitation_phytoplankton, vs.limiting_functions["phytoplankton"]))
    vs.limiting_functions["phytoplankton"] += [maximized_dop_po4_limitation_phytoplankton]

    register_npzd_rule(vs, grazing, "diazotroph", "zooplankton", label="Grazing")
    register_npzd_rule(vs, recycling_to_po4, "diazotroph", "po4", label="Fast recycling")
    register_npzd_rule(vs, recycling_to_no3, "diazotroph", "no3", label="Fast recycling")
    register_npzd_rule(vs, empty_rule, "diazotroph", "DON", label="Fast recycling")
    register_npzd_rule(vs, empty_rule, "diazotroph", "DOP", label="Fast recycling")
    register_npzd_rule(vs, primary_production_from_po4_dop, "po4", "diazotroph", label="Primary production")
    register_npzd_rule(vs, empty_rule, "no3", "diazotroph", label="Primary production")
    register_npzd_rule(vs, primary_production_from_dop_po4, "DOP", "diazotroph", label="Primary production")
    register_npzd_rule(vs, empty_rule, "DON", "diazotroph", label="Primary production")
    register_npzd_rule(vs, mortality, "diazotroph", "detritus", label="Mortality")
    register_npzd_rule(vs, recycling_to_no3, "detritus", "no3", label="Remineralization")
    register_npzd_rule(vs, excretion, "zooplankton", "no3", label="Excretion")
    register_npzd_rule(vs, empty_rule, "DOP", "po4", label="Remineralization??")
    register_npzd_rule(vs, empty_rule, "DON", "no3", label="Remineralization??")
    register_npzd_rule(vs, primary_production_from_dop_po4, "DOP", "phytoplankton", label="Primary production")
    register_npzd_rule(vs, primary_production_from_po4_dop, "po4", "phytoplankton", label="Primary production")


@veros_method
def setup_calcifying_npzd_rules(vs):
    """
    Rules for calcifying coccolitophores and caco3 tracking
    """
    # TODO: complete rules: Should be trivial if nitrogen is working
    from .npzd_rules import primary_production, recycling_to_po4, mortality, grazing,\
        recycling_phyto_to_dic, primary_production_from_DIC  # , calpro

    vs.zprefs["coccolitophore"] = vs.zprefC
    vs.plankton_types.append("coccolitophore")

    vs.mortality_rates["coccolitophore"] = vs.specific_mortality_coccolitophore
    vs.recycling_rates["coccolitophore"] = vs.nuct0

    vs.plankton_growth_functions["coccolitophore"] = coccolitophore_potential_growth
    vs.limiting_functions["coccolitophore"] = [phosphate_limitation_coccolitophore]

    vs.sinking_speeds["caco3"] = (vs.wc0 + vs.mw_c * np.where(-vs.zw < vs.mwz, -vs.zw, vs.mwz))\
        * vs.maskT

    register_npzd_data(vs, "caco3", vs.caco3)
    register_npzd_data(vs, "coccolitophore", vs.coccolitophore)

    register_npzd_rule(vs, primary_production, "po4", "coccolitophore", label="Primary production")
    register_npzd_rule(vs, recycling_to_po4, "coccolitophore", "po4", label="Fast recycling")
    register_npzd_rule(vs, mortality, "coccolitophore", "detritus", label="Mortality")
    register_npzd_rule(vs, recycling_phyto_to_dic, "coccolitophore", "DIC", label="Fast recycling")
    register_npzd_rule(vs, primary_production_from_DIC, "DIC", "coccolitophore", label="Primary production")
    register_npzd_rule(vs, grazing, "coccolitophore", "zooplankton", label="Grazing")

    # TODO add calcifying rules.
    # register_npzd_rule(vs, calpro, "coccolitophore", "caco3", label="Calcite production due to sloppy feeding???")
    # register_npzd_rule(vs, calpro, "zooplankton", "caco3", label="Calcite production due to sloppy feeding???")


@veros_method
def setupNPZD(vs):
    """Taking veros variables and packaging them up into iterables"""

    # TODO can we create the dictionaries in variables or something like that?
    vs.npzd_tracers = {}  # Dictionary keeping track of plankton, nutrients etc.
    vs.npzd_rules = []  # List of rules describing the interaction between tracers
    vs.npzd_pre_rules = []
    vs.npzd_post_rules = []


    # Temporary storage of mortality and recycled - to be used in rules
    # TODO make these local somehow or prefarably produce the results directly in the rules
    vs.net_primary_production = {}
    vs.recycled = {}
    vs.mortality = {}

    # TODO move elsewhere - maybe diagnostics
    if vs.show_npzd_graph:
        from graphviz import Digraph
        # graph for visualizing interactions - usefull for debugging
        vs.npzd_graph = Digraph("npzd_dynamics", filename="npzd_dynamics.gv", format="png")

        vs.npzd_graph.graph_attr['splines'] = 'ortho'
        vs.npzd_graph.graph_attr['nodesep'] = '1'

    vs.plankton_growth_functions = {}  # Contains functions describing growth of plankton
    vs.limiting_functions = {}  # Contains descriptions of how nutrients put a limit on growth

    vs.sinking_speeds = {}  # Dictionary of sinking objects with their sinking speeds
    vs.recycling_rates = {}
    vs.mortality_rates = {}

    setup_basic_npzd_rules(vs)

    # Add carbon to the model
    if vs.enable_carbon:
        setup_carbon_npzd_rules(vs)

    # Add nitrogen cycling to the model
    if vs.enable_nitrogen:
        setup_nitrogen_npzd_rules(vs)

    # Add calcifying coccolitophores and explicit tracking of caco3
    if vs.enable_calcifiers:
        setup_calcifying_npzd_rules(vs)

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

    # Keep derivatives of everything for advection
    vs.npzd_tracer_derivatives = {tracer: np.zeros_like(data)
                                  for tracer, data in vs.npzd_tracers.items()}

    # Whether or not to display a graph of the intended dynamics to the user
    # TODO move this to diagnostics or something
    if vs.show_npzd_graph:
        vs.npzd_graph.view()


@veros_method
def npzd(vs):
    """
    Main driving function for NPZD functionality
    Computes transport terms and biological activity separately

    :math: \\dfrac{\\partial C_i}{\\partial t} = T + S
    """

    # TODO: Refactor transportation code to be defined only once and also used by thermodynamics
    # TODO: Dissipation on W-grid if necessary

    npzd_changes = biogeochemistry(vs)

    # # TODO if this is called by thermodynamics first, then we don't have to do it again
    if vs.enable_neutral_diffusion:
        isoneutral.isoneutral_diffusion_pre(vs)


    """
    For vertical mixing
    """

    # TODO: move to function. This is essentially the same as vmix in thermodynamics
    a_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
    b_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
    c_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
    d_tri = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)
    delta = np.zeros((vs.nx, vs.ny, vs.nz), dtype=vs.default_float_type)

    ks = vs.kbot[2:-2, 2:-2] - 1
    delta[:, :, :-1] = vs.dt_tracer / vs.dzw[np.newaxis, np.newaxis, :-1]\
        * vs.kappaH[2:-2, 2:-2, :-1]
    delta[:, :, -1] = 0
    a_tri[:, :, 1:] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:]
    b_tri[:, :, 1:] = 1 + (delta[:, :, 1:] + delta[:, :, :-1]) / vs.dzt[np.newaxis, np.newaxis, 1:]
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1]


    for tracer, tracer_data in vs.npzd_tracers.items():
        vs.npzd_tracers[tracer][:, :, :, vs.taup1] = vs.npzd_tracers[tracer][:, :, :, vs.tau]

        """
        Advection of tracers
        """
        # TODO: Rename npzd_tracer_derivatives to npzd_advection_derivates? or something like that
        thermodynamics.advect_tracer(vs, tracer_data[:, :, :, vs.tau],
                                     vs.npzd_tracer_derivatives[tracer][:, :, :, vs.tau])

        # Adam-Bashforth timestepping
        # TODO I might not need 3 timestep values of the tracers. Just the derivatives
        tracer_data[:, :, :, vs.taup1] = tracer_data[:, :, :, vs.tau] + vs.dt_tracer \
            * ((1.5 + vs.AB_eps) * vs.npzd_tracer_derivatives[tracer][:, :, :, vs.tau]
               - (0.5 + vs.AB_eps) * vs.npzd_tracer_derivatives[tracer][:, :, :, vs.taum1])\
            * vs.maskT

        """
        Diffusion of tracers
        """

        if vs.enable_hor_diffusion:
            horizontal_diffusion_change = np.zeros_like(tracer_data[:, :, :, 0])
            diffusion.horizontal_diffusion(vs, tracer_data[:, :, :, vs.tau], horizontal_diffusion_change)

            tracer_data[:, :, :, vs.taup1] += vs.dt_tracer * horizontal_diffusion_change

        if vs.enable_biharmonic_mixing:
            biharmonic_diffusion_change = np.empty_like(tracer_data[:, :, :, 0])
            diffusion.biharmonic(vs, tracer_data[:, :, :, vs.tau],
                                 np.sqrt(abs(vs.K_hbi)), biharmonic_diffusion_change)

            tracer_data[:, :, :, vs.taup1] += vs.dt_tracer * biharmonic_diffusion_change

        """
        Restoring zones
        """
        # TODO add restoring zones to general tracers


        """
        Isopycnal diffusion
        """

        if vs.enable_neutral_diffusion:
            dtracer_iso = np.zeros_like(tracer_data[..., 0])

            # NOTE isoneutral_diffusion_decoupled is a temporary solution to splitting the explicit
            # dependence on time and salinity from the function isoneutral_diffusion
            isoneutral.isoneutral_diffusion_decoupled(vs, tracer_data, dtracer_iso,
                                                      iso=True, skew=False)

            if vs.enable_skew_diffusion:
                dtracer_skew = np.zeros_like(tracer_data[..., 0])
                isoneutral.isoneutral_diffusion_decoupled(vs, tracer_data, dtracer_skew,
                                                          iso=False, skew=True)


        """
        Vertical mixing of tracers
        """
        d_tri[:, :, :] = tracer_data[2:-2, 2:-2, :, vs.taup1]
        # TODO: surface flux?
        # d_tri[:, :, -1] += surface_forcing
        sol, mask = utilities.solve_implicit(vs, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)

        tracer_data[2:-2, 2:-2, :, vs.taup1] = utilities.where(vs, mask, sol,
                                                               tracer_data[2:-2, 2:-2, :, vs.taup1])

    for tracer, change in npzd_changes.items():
        vs.npzd_tracers[tracer][:, :, :, vs.taup1] += change

    for tracer in vs.npzd_tracers.values():
        tracer[:, :, :, vs.taup1] = np.maximum(tracer[:, :, :, vs.taup1], vs.trcmin * vs.maskT)

    if vs.enable_cyclic_x:
        for tracer in vs.npzd_tracers.values():
            cyclic.setcyclic_x(tracer)
