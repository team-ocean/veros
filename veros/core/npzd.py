"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
from collections import namedtuple
import numpy as np  # NOTE np is already defined by Veros and should be removed
from .. import veros_method, veros_inline_method
from .. import time
from . import diffusion, thermodynamics, cyclic, utilities, isoneutral


@veros_method
def biogeochemistry(vs):
    """
    Control function for integration of biogeochemistry

    Implements a rule based strategy. Any interaction between tracers should be
    described by registering a rule for the interaction.
    This routine ensures minimum tracer concentrations,
    calculates primary production, mortality, recyclability, vertical export for
    general tracers. Tracers should be registered to be used.

    TODO: More description or link to documentation
    """

    # Number of timesteps to do for bio tracers
    nbio = int(vs.dt_tracer // vs.dt_bio)

    # temporary tracer object to store differences
    for tracer, val in vs.npzd_tracers.items():
        vs.temporary_tracers[tracer][:, :, :] = val[:, :, :, vs.tau]

    # Flags enable us to only work on tracers with a minimum available concentration
    flags = {tracer: vs.maskT[...].astype(np.bool) for tracer in vs.temporary_tracers}

    # Pre rules: Changes that need to be applied before running npzd dynamics
    pre_rules = [(rule.function(vs, rule.source, rule.sink), rule.boundary)
                 for rule in vs.npzd_pre_rules]

    for rule, boundary in pre_rules:
        for key, value in rule.items():
            vs.temporary_tracers[key][boundary] += value

    # How much plankton is blocking light
    plankton_total = sum([vs.temporary_tracers[plankton] for plankton
                          in vs.plankton_types]) * vs.dzt

    # Integrated phytplankton - starting from top of layer going upwards
    # reverse cumulative sum because our top layer is the last.
    # Needs to be reversed again to reflect direction
    phyto_integrated = np.empty_like(vs.temporary_tracers['phytoplankton'])
    phyto_integrated[:, :, :-1] = plankton_total[:, :, 1:]
    phyto_integrated[:, :, -1] = 0.0

    # incomming shortwave radiation at top of layer
    swr = vs.swr[:, :, np.newaxis] * \
          np.exp(-vs.light_attenuation_phytoplankton
                 * np.cumsum(phyto_integrated[:, :, ::-1], axis=2)[:, :, ::-1])

    # Reduce incomming light where there is ice - as veros doesn't currently
    # have an ice model, we get temperatures below -1.8 and decreasing temperature forcing
    # as recommended by the 4deg model from the setup gallery
    icemask = np.logical_and(vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8, vs.forc_temp_surface < 0.0)
    swr[:, :] *= np.exp(-vs.light_attenuation_ice * icemask[:, :, np.newaxis])

    # declination and fraction of day with daylight
    # 0.72 is fraction of year at aphelion
    # 0.4 is scaling based on angle of rotation
    declin = np.sin((np.mod(vs.time * time.SECONDS_TO_X['years'], 1) - 0.72) * 2.0 * np.pi) * 0.4
    radian = 2 * np.pi / 360  # bohrium doesn't support np.radians, so this is faster
    rctheta = np.maximum(-1.5, np.minimum(1.5, vs.yt * radian - declin))

    # 1.33 is derived from Snells law for the air-sea barrier
    vs.rctheta[:] = vs.light_attenuation_water / np.sqrt(1.0 - (1.0 - np.cos(rctheta)**2.0) / 1.33**2)

    # fraction of day with photosynthetically active radiation with a minimum value
    dayfrac = np.minimum(1.0, -np.tan(radian * vs.yt) * np.tan(declin))
    vs.dayfrac[:] = np.maximum(1e-12, np.arccos(np.maximum(-1.0, dayfrac)) / np.pi)

    # light at top of grid box
    grid_light = swr * np.exp(vs.zw[np.newaxis, np.newaxis, :]
                              * vs.rctheta[np.newaxis, :, np.newaxis])

    # amount of PAR absorbed by water and plankton in each grid cell
    light_attenuation = vs.dzt * vs.light_attenuation_water +\
                        plankton_total * vs.light_attenuation_phytoplankton

    # recycling rate determined according to b ** (cT)
    bct = vs.bbio ** (vs.cbio * vs.temp[:, :, :, vs.tau])
    bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, :, vs.tau],
                                            vs.zooplankton_max_growth_temp))

    # Maximum grazing rate is a function of temperature
    # bctz sets an upper limit on effects of temperature on grazing
    # gmax = vs.gbio * bctz
    gmax = vs.maximum_grazing_rate * bctz

    # light saturated growth and non-saturated growth
    jmax, avej = {}, {}
    for plankton, growth_function in vs.plankton_growth_functions.items():
        jmax[plankton], avej[plankton] = growth_function(vs, bct, grid_light, light_attenuation)

    # bio loop
    for _ in range(nbio):

        # Plankton is recycled, dying and growing
        # pre compute amounts for use in rules
        for plankton in vs.plankton_types:

            # Nutrient limiting growth - if no limit, growth is determined by avej
            u = 1

            # limit maximum growth, usually by nutrient deficiency
            for growth_limiting_function in vs.limiting_functions[plankton]:
                u = np.minimum(u, growth_limiting_function(vs, vs.temporary_tracers))

            # increase in plankton population by nutrient consumption
            vs.net_primary_production[plankton] = flags[plankton] * flags['po4']\
                * np.minimum(avej[plankton], u * jmax[plankton]) * vs.temporary_tracers[plankton]

            # Fast recycling of plankton
            vs.recycled[plankton] = flags[plankton] * vs.recycling_rates[plankton] * bct\
                * vs.temporary_tracers[plankton]

            # Mortality of plankton
            vs.mortality[plankton] = flags[plankton] * vs.mortality_rates[plankton]\
                * vs.temporary_tracers[plankton]

        # Detritus is recycled
        vs.recycled['detritus'] = flags['detritus'] * vs.recycling_rates['detritus'] * bct\
            * vs.temporary_tracers['detritus']

        # zooplankton displays quadric mortality rates
        vs.mortality['zooplankton'] = flags['zooplankton'] * vs.quadric_mortality_zooplankton\
            * vs.temporary_tracers['zooplankton'] ** 2

        # Zooplankton growth parameters for use in rules and diagnostics
        vs.grazing, vs.digestion, vs.excretion, vs.sloppy_feeding = \
            zooplankton_grazing(vs, vs.temporary_tracers, flags, gmax)
        vs.excretion_total[...] = sum(vs.excretion.values())

        # Calculate concentration leaving cell and entering from above
        for sinker, speed in vs.sinking_speeds.items():
            # Concentration of exported material is calculated as fraction
            # of total concentration which would have fallen through the bottom
            # of the cell (speed / vs.dzt * vs.dtbio)
            # vs.dtbio is accounted for later
            vs.npzd_export[sinker] = speed / vs.dzt * vs.temporary_tracers[sinker] * flags[sinker]

            # Import is export from above scaled by the ratio of cell heights
            vs.npzd_import[sinker] = np.empty_like(vs.npzd_export[sinker])
            vs.npzd_import[sinker][:, :, -1] = 0
            vs.npzd_import[sinker][:, :, :-1] = vs.npzd_export[sinker][:, :, 1:] * (vs.dzt[1:] /
                                                                                    vs.dzt[:-1])
            vs.npzd_import[sinker][...] *= vs.maskT

        # Gather all state updates
        npzd_updates = [(rule.function(vs, rule.source, rule.sink), rule.boundary)
                        for rule in vs.npzd_rules]

        # perform updates
        for update, boundary in npzd_updates:
            for key, value in update.items():
                vs.temporary_tracers[key][boundary] += value * vs.dt_bio

        # Import and export between layers
        for tracer in vs.sinking_speeds:
            vs.temporary_tracers[tracer][:, :, :] += \
                            (vs.npzd_import[tracer] - vs.npzd_export[tracer]) * vs.dt_bio

        # Prepare temporary tracers for next bio iteration
        for tracer, data in vs.temporary_tracers.items():
            flags[tracer][:, :, :] = np.logical_and(flags[tracer], (data > vs.trcmin))
            data[:, :, :] = utilities.where(vs, flags[tracer], data, vs.trcmin)

    # Post processesing or smoothing rules
    post_results = [(rule.function(vs, rule.source, rule.sink), rule.boundary)
                    for rule in vs.npzd_post_rules]
    post_modified = []  # we only want to reset values, which have acutally changed for performance

    for result, boundary in post_results:
        for key, value in result.items():
            vs.temporary_tracers[key][boundary] += value
            post_modified.append(key)

    # Reset before returning
    # using set for unique modifications is faster than resetting all tracers
    for tracer in set(post_modified):
        data = vs.temporary_tracers[tracer]
        flags[tracer][:, :, :] = np.logical_and(flags[tracer], (data > vs.trcmin))
        data[:, :, :] = utilities.where(vs, flags[tracer], data, vs.trcmin)

    # Only return the difference from the current time step. Will be added to timestep taup1
    return {tracer: vs.temporary_tracers[tracer] - vs.npzd_tracers[tracer][:, :, :, vs.tau]
            for tracer in vs.npzd_tracers}


@veros_inline_method
def zooplankton_grazing(vs, tracers, flags, gmax):
    """
    Zooplankton grazing returns total grazing, digestion i.e. how much is available
    for zooplankton growth, excretion and sloppy feeding
    All are useful to have calculated once and made available to rules
    """

    # Denominator
    thetaZ = sum([pref_score * tracers[preference] for preference, pref_score
                  in vs.zprefs.items()])\
                 + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

    # Fractional part of each grazed type, which may be ingested
    ingestion = {preference: pref_score / thetaZ for preference, pref_score in vs.zprefs.items()}

    # The amounts each grazed tracer will be reduced
    grazing = {preference: flags[preference] * flags['zooplankton'] * gmax
                           * ingestion[preference] * tracers[preference]
                           * tracers['zooplankton'] for preference in ingestion}

    # how much is available for growth in zooplankton
    digestion = {preference: vs.assimilation_efficiency * amount_grazed
                 for preference, amount_grazed in grazing.items()}

    # how much is lost from zooplankton growth, based on digestion
    excretion = {preference: (1 - vs.zooplankton_growth_efficiency) * amount_digested
                 for preference, amount_digested in digestion.items()}

    # the part of total grazed tracers which are not digested - i.e. it should become detritus
    sloppy_feeding = {preference: grazing[preference] - digestion[preference]
                      for preference in grazing}

    return grazing, digestion, excretion, sloppy_feeding


@veros_inline_method
def potential_growth(vs, bct, grid_light, light_attenuation, growth_parameter):
    """ Potential growth of phytoplankton """
    f1 = np.exp(-light_attenuation)  # available light
    jmax = growth_parameter * bct  # maximum growth
    gd = jmax * vs.dayfrac[np.newaxis, :, np.newaxis]  # growth in fraction of day
    avej = avg_J(vs, f1, gd, grid_light, light_attenuation)  # light limited growth

    return jmax, avej


@veros_inline_method
def phytoplankton_potential_growth(vs, bct, grid_light, light_attenuation):
    """ Regular potential growth scaled by vs.abi_P """
    # return potential_growth(vs, bct, grid_light, light_attenuation, vs.abio_P)
    return potential_growth(vs, bct, grid_light, light_attenuation, vs.maximum_growth_rate_phyto)


@veros_inline_method
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


@veros_inline_method
def phosphate_limitation_phytoplankton(vs, tracers):
    """ Phytoplankton limit to growth by phosphate limitation """
    return general_nutrient_limitation(tracers["po4"], vs.saturation_constant_N * vs.redfield_ratio_PN)


@veros_inline_method
def register_npzd_data(vs, name, value, transport=True):
    """
    Add tracer to the NPZD data set and create node in interaction graph
    Tracers added are available in the npzd dynamics and is automatically
    included in transport equations
    """

    if name in vs.npzd_tracers:
        raise ValueError(name, "has already been added to the NPZD data set")

    vs.npzd_tracers[name] = value

    if transport:
        vs.npzd_transported_tracers.append(name)


@veros_inline_method
def _get_boundary(vs, boundary_string):
    """
    Return slice representing boundary

    surface:       [:, :, -1] only the top layer
    bottom:        bottom_mask as set by veros
    else:          [:, :, :] everything
    """

    if boundary_string == 'SURFACE':
        return tuple([slice(None, None, None), slice(None, None, None), -1])

    if boundary_string == 'BOTTOM':
        return vs.bottom_mask

    return tuple([slice(None, None, None)] * 3)


@veros_inline_method
def register_npzd_rule(vs, name, rule, label=None, boundary=None, group="PRIMARY"):
    """ Add rule to the npzd dynamics e.g. phytoplankkton being eaten by zooplankton

        ...
        name: Unique identifier for the rule
        rule: a list of rule names or tuple containing:
            function: function to be called
            source: what is being consumed
            destination: what is growing from consuming
        label: A description for graph
        boundary: 'SURFACE', 'BOTTOM' or None
        ...
    """

    Rule = namedtuple('Rule', ['function', 'source', 'sink', 'label', 'boundary', 'group'])

    if name in vs.npzd_available_rules:
        raise ValueError('Rule %s already exists, please verify the rule has not already been added and replace the chosen name' % name)

    if isinstance(rule, list):
        if label or boundary:
            raise ValueError('Cannot add labels or boundaries to rule groups')
        vs.npzd_available_rules[name] = rule

    else:
        label = label or '?'  # label is just for the interaction graph
        vs.npzd_available_rules[name] = Rule(function=rule[0], source=rule[1],
                                             sink=rule[2], label=label,
                                             boundary=_get_boundary(vs, boundary),
                                             group=group)


@veros_inline_method
def select_npzd_rule(vs, name):
    """ Select rule for the NPZD model """

    # activate rule by selecting from available rules
    rule = vs.npzd_available_rules[name]
    if name in vs.npzd_selected_rule_names:
        raise ValueError('Rules must have unique names, %s defined multiple times' % name)

    vs.npzd_selected_rule_names.append(name)

    # we may activate each rule in a list of rules
    if isinstance(rule, list):
        for r in rule:
            select_npzd_rule(vs, r)

    # or activate a single rule
    elif isinstance(rule, tuple):

        if rule.group == 'PRIMARY':
            vs.npzd_rules.append(rule)
        elif rule.group == 'PRE':
            vs.npzd_pre_rules.append(rule)
        elif rule.group == 'POST':
            vs.npzd_post_rules.append(rule)

    else:
        raise TypeError('Rule must be of type tuple or list')


@veros_inline_method
def setup_basic_npzd_rules(vs):
    """
    Setup rules for basic NPZD model including phosphate, detritus, phytoplankton and zooplankton
    """
    from .npzd_rules import grazing, mortality, sloppy_feeding, recycling_to_po4, \
        zooplankton_self_grazing, excretion, primary_production, empty_rule, \
        bottom_remineralization_detritus_po4

    # TODO - couldn't this be created elsewhere or can I use vs.kbot more efficiently?
    vs.bottom_mask = np.empty((vs.nx + 4, vs.ny + 4, vs.nz), dtype=np.bool)
    vs.bottom_mask[:, :, :] = np.arange(vs.nz)[np.newaxis, np.newaxis, :] == (vs.kbot - 1)[:, :, np.newaxis]

    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
    vs.sinking_speeds['detritus'] = (vs.wd0 + vs.mw * np.where(-zw < vs.mwz, -zw, vs.mwz)) \
        * vs.maskT

    # Add "regular" phytoplankton to the model
    vs.plankton_types = ['phytoplankton']  # Phytoplankton types in the model. For blocking light
    vs.plankton_growth_functions['phytoplankton'] = phytoplankton_potential_growth
    vs.limiting_functions['phytoplankton'] = [phosphate_limitation_phytoplankton]
    # vs.recycling_rates['phytoplankton'] = vs.nupt0
    vs.recycling_rates['phytoplankton'] = vs.fast_recycling_rate_phytoplankton
    # vs.recycling_rates['detritus'] = vs.nud0
    vs.recycling_rates['detritus'] = vs.remineralization_rate_detritus
    vs.mortality_rates['phytoplankton'] = vs.specific_mortality_phytoplankton

    # Zooplankton preferences for grazing on keys
    # Values are scaled automatically at the end of this function
    # vs.zprefs = {'phytoplankton': vs.zprefP, 'zooplankton': vs.zprefZ, 'detritus': vs.zprefDet}
    vs.zprefs['phytoplankton'] = vs.zprefP
    vs.zprefs['zooplankton'] = vs.zprefZ
    vs.zprefs['detritus'] = vs.zprefDet

    # Register for basic model
    register_npzd_data(vs, 'detritus', vs.detritus)
    register_npzd_data(vs, 'phytoplankton', vs.phytoplankton)
    register_npzd_data(vs, 'zooplankton', vs.zooplankton)
    register_npzd_data(vs, 'po4', vs.po4)

    # Register rules for interactions between active tracers
    register_npzd_rule(vs, 'npzd_basic_phytoplankton_grazing',
                       (grazing, 'phytoplankton', 'zooplankton'),
                       label='Grazing')
    register_npzd_rule(vs, 'npzd_basic_phytoplankton_mortality',
                       (mortality, 'phytoplankton', 'detritus'),
                       label='Mortality')
    register_npzd_rule(vs, 'npzd_basic_phytoplankton_sloppy_feeding',
                       (sloppy_feeding, 'phytoplankton', 'detritus'),
                       label='Sloppy feeding')
    register_npzd_rule(vs, 'npzd_basic_phytoplankton_fast_recycling',
                       (recycling_to_po4, 'phytoplankton', 'po4'),
                       label='Fast recycling')
    register_npzd_rule(vs, 'npzd_basic_zooplankton_grazing',
                       (zooplankton_self_grazing, 'zooplankton', 'zooplankton'),
                       label='Grazing')
    register_npzd_rule(vs, 'npzd_basic_zooplankton_excretion',
                       (excretion, 'zooplankton', 'po4'),
                       label='Excretion')
    register_npzd_rule(vs, 'npzd_basic_zooplankton_mortality',
                       (mortality, 'zooplankton', 'detritus'),
                       label='Mortality')
    register_npzd_rule(vs, 'npzd_basic_zooplankton_sloppy_feeding',
                       (sloppy_feeding, 'zooplankton', 'detritus'),
                       label='Sloppy feeding')
    register_npzd_rule(vs, 'npzd_basic_detritus_sloppy_feeding',
                       (sloppy_feeding, 'detritus', 'detritus'),
                       label='Sloppy feeding')
    register_npzd_rule(vs, 'npzd_basic_detritus_grazing',
                       (grazing, 'detritus', 'zooplankton'),
                       label='Grazing')
    register_npzd_rule(vs, 'npzd_basic_detritus_remineralization',
                       (recycling_to_po4, 'detritus', 'po4'),
                       label='Remineralization')
    register_npzd_rule(vs, 'npzd_basic_phytoplankton_primary_production',
                       (primary_production, 'po4', 'phytoplankton'),
                       label='Primary production')

    register_npzd_rule(vs, 'npzd_basic_detritus_bottom_remineralization',
                       (bottom_remineralization_detritus_po4, 'detritus', 'po4'),
                       label='Bottom remineralization', boundary='BOTTOM')

    register_npzd_rule(vs, 'group_npzd_basic', [
        'npzd_basic_phytoplankton_grazing',
        'npzd_basic_phytoplankton_mortality',
        'npzd_basic_phytoplankton_sloppy_feeding',
        'npzd_basic_phytoplankton_fast_recycling',
        'npzd_basic_phytoplankton_primary_production',
        'npzd_basic_zooplankton_grazing',
        'npzd_basic_zooplankton_excretion',
        'npzd_basic_zooplankton_mortality',
        'npzd_basic_zooplankton_sloppy_feeding',
        'npzd_basic_detritus_sloppy_feeding',
        'npzd_basic_detritus_remineralization',
        'npzd_basic_detritus_grazing',
        'npzd_basic_detritus_bottom_remineralization'
        ])

    register_npzd_rule(vs, 'empty_rule', (empty_rule, None, None))


@veros_inline_method
def setup_carbon_npzd_rules(vs):
    """
    Rules for including a carbon cycle
    """
    # The actual action is on DIC, but the to variables overlap
    from .npzd_rules import co2_surface_flux, recycling_to_dic, \
        primary_production_from_DIC, excretion_dic, recycling_phyto_to_dic, \
        dic_alk_scale, calcite_production_phyto, calcite_production_phyto_alk, \
        post_redistribute_calcite, post_redistribute_calcite_alk, pre_reset_calcite, \
        bottom_remineralization_detritus_DIC

    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird

    # redistribution fraction for calcite at level k
    vs.rcak[:, :, :-1] = (- np.exp(zw[:-1] / vs.dcaco3) + np.exp(zw[1:] / vs.dcaco3)) / vs.dzt[:-1]
    vs.rcak[:, :, -1] = - (np.exp(zw[-1] / vs.dcaco3) - 1.0) / vs.dzt[-1]

    # redistribution fraction at bottom
    rcab = np.empty_like(vs.DIC[..., 0])
    rcab[:, : -1] = 1 / vs.dzt[-1]
    rcab[:, :, :-1] = np.exp(zw[:-1] / vs.dcaco3) / vs.dzt[1:]

    # merge bottom into level k and reset every cell outside ocean
    vs.rcak[vs.bottom_mask] = rcab[vs.bottom_mask]
    vs.rcak[...] *= vs.maskT

    # Need to track dissolved inorganic carbon, alkalinity
    register_npzd_data(vs, 'DIC', vs.DIC)
    register_npzd_data(vs, 'alkalinity', vs.alkalinity)

    if not vs.enable_calcifiers:
        # Only for collection purposes - to be redistributed in post rules
        register_npzd_data(vs, 'caco3', np.zeros_like(vs.DIC), transport=False)

    # Atmosphere
    register_npzd_rule(vs, 'npzd_carbon_flux',
                       (co2_surface_flux, 'co2', 'DIC'),
                       boundary='SURFACE', group='PRE')

    # Common rule set for nutrient
    register_npzd_rule(vs, 'npzd_carbon_recycling_detritus_dic',
                       (recycling_to_dic, 'detritus', 'DIC'),
                       label='Remineralization')
    register_npzd_rule(vs, 'npzd_carbon_primary_production_dic',
                       (primary_production_from_DIC, 'DIC', 'phytoplankton'),
                       label='Primary production')
    register_npzd_rule(vs, 'npzd_carbon_recycling_phyto_dic',
                       (recycling_phyto_to_dic, 'phytoplankton', 'DIC'),
                       label='Fast recycling')
    register_npzd_rule(vs, 'npzd_carbon_excretion_dic',
                       (excretion_dic, 'zooplankton', 'DIC'),
                       label='Excretion')
    register_npzd_rule(vs, 'npzd_carbon_calcite_production_dic',
                       (calcite_production_phyto, 'DIC', 'caco3'),
                       label='Production of calcite')
    register_npzd_rule(vs, 'npzd_carbon_calcite_production_alk',
                       (calcite_production_phyto_alk, 'alkalinity', 'caco3'),
                       label='Production of calcite')

    # BOTTOM
    register_npzd_rule(vs, 'npzd_carbon_detritus_bottom_remineralization',
                       (bottom_remineralization_detritus_DIC, 'detritus', 'DIC'),
                       label='Bottom remineralization', boundary='BOTTOM')

    # POST rules
    register_npzd_rule(vs, 'npzd_carbon_dic_alk',
                       (dic_alk_scale, 'DIC', 'alkalinity'),
                       group='POST')
    register_npzd_rule(vs, 'npzd_carbon_post_distribute_calcite_alk',
                       (post_redistribute_calcite_alk, 'caco3', 'alkalinity'),
                       label='dissolution', group='POST')
    register_npzd_rule(vs, 'npzd_carbon_post_distribute_calcite_dic',
                       (post_redistribute_calcite, 'caco3', 'DIC'),
                       label='dissolution', group='POST')

    # PRE
    register_npzd_rule(vs, 'pre_reset_calcite',
                       (pre_reset_calcite, 'caco3', 'caco3'),
                       label='reset', group='PRE')

    register_npzd_rule(vs, 'group_carbon_implicit_caco3', [
        'npzd_carbon_flux',
        'npzd_carbon_recycling_detritus_dic',
        'npzd_carbon_primary_production_dic',
        'npzd_carbon_recycling_phyto_dic',
        'npzd_carbon_excretion_dic',
        'npzd_carbon_dic_alk',
        'npzd_carbon_calcite_production_dic',
        'npzd_carbon_calcite_production_alk',
        'npzd_carbon_post_distribute_calcite_alk',
        'npzd_carbon_post_distribute_calcite_dic',
        'npzd_carbon_detritus_bottom_remineralization',
        'pre_reset_calcite',
        ])


@veros_inline_method
def setupNPZD(vs):
    """Taking veros variables and packaging them up into iterables"""

    # Which tracers should be transported
    # In some cases it may be desirable to not transport a tracer. In that
    # case you should ensure, it is updated or reset appropriately using pre and post rules
    # vs.npzd_transported_tracers = []

    setup_basic_npzd_rules(vs)

    # Add carbon to the model
    if vs.enable_carbon:
        setup_carbon_npzd_rules(vs)

    for rule in vs.npzd_selected_rules:
        select_npzd_rule(vs, rule)

    # Update Zooplankton preferences dynamically
    zprefsum = sum(vs.zprefs.values())
    for preference in vs.zprefs:
        vs.zprefs[preference] /= zprefsum

    # Keep derivatives of everything for advection
    for tracer, data in vs.npzd_tracers.items():
        vs.npzd_advection_derivatives[tracer] = np.zeros_like(data)

    # Temporary tracers are necessary to only return differences
    for tracer, data in vs.npzd_tracers.items():
        vs.temporary_tracers[tracer] = np.empty_like(data[..., 0])


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

    for tracer in vs.npzd_transported_tracers:
        tracer_data = vs.npzd_tracers[tracer]

        """
        Advection of tracers
        """
        thermodynamics.advect_tracer(vs, tracer_data[:, :, :, vs.tau],
                                     vs.npzd_advection_derivatives[tracer][:, :, :, vs.tau])

        # Adam-Bashforth timestepping
        tracer_data[:, :, :, vs.taup1] = tracer_data[:, :, :, vs.tau] + vs.dt_tracer \
            * ((1.5 + vs.AB_eps) * vs.npzd_advection_derivatives[tracer][:, :, :, vs.tau]
               - (0.5 + vs.AB_eps) * vs.npzd_advection_derivatives[tracer][:, :, :, vs.taum1])\
            * vs.maskT

        """
        Diffusion of tracers
        """

        if vs.enable_hor_diffusion:
            horizontal_diffusion_change = np.zeros_like(tracer_data[:, :, :, 0])
            diffusion.horizontal_diffusion(vs, tracer_data[:, :, :, vs.tau],
                                           horizontal_diffusion_change)

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

    # update by biogeochemical changes
    for tracer, change in npzd_changes.items():
        vs.npzd_tracers[tracer][:, :, :, vs.taup1] += change

    # prepare next timestep with minimum tracer values
    for tracer in vs.npzd_tracers.values():
        tracer[:, :, :, vs.taup1] = np.maximum(tracer[:, :, :, vs.taup1], vs.trcmin * vs.maskT)

    if vs.enable_cyclic_x:
        for tracer in vs.npzd_tracers.values():
            cyclic.setcyclic_x(tracer)

