"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
from collections import namedtuple

from .. import veros_method
from .. import time
from . import diffusion, thermodynamics, utilities, isoneutral
from ..variables import allocate


@veros_method
def biogeochemistry(vs):
    """Control function for integration of biogeochemistry

    Implements a rule based strategy. Any interaction between tracers should be
    described by registering a rule for the interaction.
    This routine ensures minimum tracer concentrations,
    calculates primary production, mortality, recyclability, vertical export for
    general tracers. Tracers should be registered to be used.
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
    plankton_total = sum([plankton for plankton in vs.temporary_tracers.values()
                          if hasattr(plankton, 'light_attenuation')]) * vs.dzt

    # Integrated phytplankton - starting from top of layer going upwards
    # reverse cumulative sum because our top layer is the last.
    # Needs to be reversed again to reflect direction
    # phyto_integrated = np.empty_like(vs.temporary_tracers['phytoplankton'])
    phyto_integrated = np.empty_like(plankton_total)
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

    # common temperature factor determined according to b ** (cT)
    vs.bct = vs.bbio ** (vs.cbio * vs.temp[:, :, :, vs.tau])

    # light saturated growth and non-saturated growth
    jmax, avej = {}, {}
    for tracer in vs.temporary_tracers.values():

        # Calculate light limited vs unlimited growth
        if hasattr(tracer, 'potential_growth'):
            jmax[tracer.name], avej[tracer.name] = tracer.potential_growth(vs, grid_light, light_attenuation)


        # Methods for internal use may need an update
        if hasattr(tracer, 'update_internal'):
            tracer.update_internal(vs)

    # bio loop
    for _ in range(nbio):

        # Plankton is recycled, dying and growing
        # pre compute amounts for use in rules
        # for plankton in vs.plankton_types:
        for tracer in vs.temporary_tracers.values():

            # Nutrient limiting growth - if no limit, growth is determined by avej
            u = 1

            # limit maximum growth, usually by nutrient deficiency
            # and calculate primary production from that
            if hasattr(tracer, 'potential_growth'):
                # NOTE jmax and avej are NOT updated within bio loop
                for growth_limiting_function in vs.limiting_functions[tracer.name]:
                    u = np.minimum(u, growth_limiting_function(vs, vs.temporary_tracers))

                vs.net_primary_production[tracer.name] = flags[tracer.name] * flags['po4'] \
                    * np.minimum(avej[tracer.name], u * jmax[tracer.name]) * tracer

            # recycling methods - remineralization and fast recycling
            if hasattr(tracer, 'recycle'):
                vs.recycled[tracer.name] = flags[tracer.name] * tracer.recycle(vs)

            # Living tracers which can die
            if hasattr(tracer, 'mortality'):
                vs.mortality[tracer.name] = flags[tracer.name] * tracer.mortality(vs)

            # tracers which can graze on others
            if hasattr(tracer, 'grazing'):
                # TODO handle grazing by multiple zooplankton types
                # currently this only works with 1 type
                vs.grazing, vs.digestion, vs.excretion, vs.sloppy_feeding = tracer.grazing(vs, vs.temporary_tracers, flags)
                vs.excretion_total[...] = sum(vs.excretion.values())

            # Calculate concentration leaving cell and entering from above
            if hasattr(tracer, 'sinking_speed'):
                # Concentration of exported material is calculated as fraction
                # of total concentration which would have fallen through the bottom
                # of the cell (speed / vs.dzt * vs.dtbio)
                # vs.dtbio is accounted for later
                vs.npzd_export[tracer.name] = tracer.sinking_speed / vs.dzt * tracer * flags[tracer.name]

                # Import is export from above scaled by the ratio of cell heights
                vs.npzd_import[tracer.name] = np.empty_like(vs.npzd_export[tracer.name])
                vs.npzd_import[tracer.name][:, :, -1] = 0
                vs.npzd_import[tracer.name][:, :, :-1] = vs.npzd_export[tracer.name][:, :, 1:] * (vs.dzt[1:] / vs.dzt[:-1])

                # ensure we don't import in cells below bottom
                vs.npzd_import[tracer.name][...] *= vs.maskT

        # Gather all state updates
        npzd_updates = [(rule.function(vs, rule.source, rule.sink), rule.boundary)
                        for rule in vs.npzd_rules]

        # perform updates
        for update, boundary in npzd_updates:
            for key, value in update.items():
                vs.temporary_tracers[key][boundary] += value * vs.dt_bio

        # Import and export between layers
        # for tracer in vs.sinking_speeds:
        for tracer in vs.temporary_tracers.values():
            if hasattr(tracer, 'sinking_speed'):
                tracer[:, :, :] += (vs.npzd_import[tracer.name] - vs.npzd_export[tracer.name]) * vs.dt_bio

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


def general_nutrient_limitation(nutrient, saturation_constant):
    """ Nutrient limitation form for all nutrients """
    return nutrient / (saturation_constant + nutrient)


@veros_method(inline=True)
def phosphate_limitation_phytoplankton(vs, tracers):
    """ Phytoplankton limit to growth by phosphate limitation """
    return general_nutrient_limitation(tracers['po4'], vs.saturation_constant_N * vs.redfield_ratio_PN)


@veros_method(inline=True)
def register_npzd_data(vs, tracer):
    """ Add tracer to the NPZD data set and create node in interaction graph

    Tracers added are available in the npzd dynamics and is automatically
    included in transport equations

    Parameters
    ----------
    tracer
        An instance of :obj:`veros.core.npzd_tracer.NPZD_tracer`
        to be included in biogeochemistry calculations
    """

    if tracer.name in vs.npzd_tracers.keys():
        raise ValueError('{name} has already been added to the NPZD data set'.format(name=tracer.name))

    vs.npzd_tracers[tracer.name] = tracer

    if tracer.transport:
        vs.npzd_transported_tracers.append(tracer.name)


@veros_method(inline=True)
def _get_boundary(vs, boundary_string):
    """ Return slice representing boundary

    Parameters
    ----------
    boundary_string
        Identifer for boundary. May take one of the following values:
        SURFACE:       [:, :, -1] only the top layer
        BOTTOM:        bottom_mask as set by veros
        else:          [:, :, :] everything
    """

    if boundary_string == 'SURFACE':
        return tuple([slice(None, None, None), slice(None, None, None), -1])

    if boundary_string == 'BOTTOM':
        return vs.bottom_mask

    return tuple([slice(None, None, None)] * 3)


@veros_method(inline=True)
def register_npzd_rule(vs, name, rule, label=None, boundary=None, group='PRIMARY'):
    """ Make rule available to the npzd dynamics

    The rule specifies an interaction between two tracers.
    It may additionally specify where in the grid it works as
    well as where in the execution order.

    Note
    ----
    Registering a rule is not sufficient for inclusion in dynamics.
    It must also be selected using select_npzd_rule.

    Parameters
    ----------
    name
        Unique identifier for the rule

    rule
        A list of rule names or tuple containing:
            function: function to be called
            source: what is being consumed
            destination: what is growing from consuming
    label
        A description for graph. See :obj:`veros.diagnostics.biogeochemistry`

    boundary
        'SURFACE', 'BOTTOM' or None, see _get_boundary

    group
        'PRIMARY' (default): Rule is evaluated in primary execution loop nbio times with timestepping
        'PRE': Rule is evaluated once before primary loop
        'POST': Rule is evaluated once after primary loop
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

@veros_method(inline=True)
def register_npzd_common_source_rule(vs, name, rule, label=None, boundary=None, group='PRIMARY'):
    """ Register a rule to the model, which shares source term with other rules.

    This function creates stub rules for each individual term by evaluating the original rule and
    returning only the sink part of the original rule. Additionally there will be created a stub
    rule for the source term based on the first registered rule for the common source. Therefore a
    common source rule is best suited for rules, where the source terms has been preevaluated.
    The new registered rules are available with the name given by the parameter :obj:`name`
    postfixed by an underscore followed by the name of the tracer to be affected as specified in the
    rule. All created rules will be made available as a rule collection to be activated collectively
    with the name specified by the parameter :obj:`name`.
    Parameters are the same as register_npzd_rule

    Note
    ----
    Should only be used for rules, where the source term is exactly the same.

    Parameters
    ----------
    name
        Unique identifier for the rule

    rule
        A list of rule names or tuple containing:
            function: function to be called
            source: what is being consumed
            destination: what is growing from consuming
    label
        A description for graph. See :obj:`veros.diagnostics.biogeochemistry`

    boundary
        'SURFACE', 'BOTTOM' or None, see _get_boundary

    group
        'PRIMARY' (default): Rule is evaluated in primary execution loop nbio times with timestepping
        'PRE': Rule is evaluated once before primary loop
        'POST': Rule is evaluated once after primary loop
    """
    # The rule collection is made in setup_npzd based on the concents of common_source_rules

    if isinstance(rule, list):
        raise ValueError('register_npzd_common_source_rule does not accept list of rule names as rule parameter')

    # This should maybe be done elsewhere
    # Ensure that the common source rules dictionary exists
    # register_nzpd_rule handles not allowing multiple rules with the same name,
    # therefore also handling not having the same sink defined twice or the source also being a sink
    if not hasattr(vs, 'common_source_rules'):
        vs.common_source_rules = {}

    # If no rule for this name has been added yet, add the source term
    if name not in vs.common_source_rules.keys():
        vs.common_source_rules[name] = []

        # Add the source term
        # The new rule should consist of a new function, which is callable like the initial function of the rule, but only returns the source term
        source_rule = (veros_method(lambda veros, source, sink: {source: rule[0](veros, source, sink)[source]}, inline=True), rule[1], rule[2])
        register_npzd_rule(vs, f'{name}_{rule[1]}', source_rule, label=label + '(source)', boundary=boundary, group=group)

        vs.common_source_rules[name].append(source_rule)

    # Register new stub rule for the sink - source is assumed common so only added for the first
    sink_rule = (veros_method(lambda veros, source, sink: {sink: rule[0](veros, source, sink)[sink]}, inline=True), rule[1], rule[2])
    register_npzd_rule(vs, f'{name}_{rule[2]}', sink_rule, label=label, boundary=boundary, group=group)

    vs.common_source_rules[name].append(sink_rule)

    # Ensure that the registered rule works on the same boundary and same execution group as the source
    # The boundary is in principle not a strict requirement, but should in practice always be the same
    registered_source = vs.npzd_available_rules[f'{name}_{rule[1]}']
    registered_sink = vs.npzd_available_rules[f'{name}_{rule[2]}']

    if registered_source.boundary != registered_sink.boundary:
        raise ValueError(f'Common source rules must have the same boundary for both source and sink but {registered_source.name} did not match {registered_sink.name}. Expected {registered_source.boundary}, got {registed_sink.boundary}')
    if registered_source.group != registered_sink.group:
        raise ValueError(f'Common source rules must have the same group for both source and sink but {registered_source.name} did not match {registered_sink.name}. Expected {registered_source.group}, got {registed_sink.group}')

    # Make sure that the sources are actually listed as the same.
    # This check should not be necesarry if the user actually wants to register rules for a common source
    if registered_source.source != rule[1]:
        raise ValueError(f'Common source rules should have a common source, but {registered_source.name} did not match {registered_sink.name}. Expected {registered_source.source}, got {rule[1]}')


@veros_method(inline=True)
def select_npzd_rule(vs, name):
    """
    Select rule for the NPZD model

    Parameters
    ----------
    name
        Name of the rule to be selected
    """

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


@veros_method(inline=True)
def setup_basic_npzd_rules(vs):
    """
    Setup rules for basic NPZD model including phosphate, detritus, phytoplankton and zooplankton
    """
    from .npzd_rules import grazing, mortality, sloppy_feeding, recycling_to_po4, \
        primary_production, empty_rule, \
        bottom_remineralization_detritus_po4

    from .npzd_tracers import Recyclable_tracer, Phytoplankton, Zooplankton, NPZD_tracer

    # TODO - couldn't this be created elsewhere or can I use vs.kbot more efficiently?
    vs.bottom_mask[:, :, :] = np.arange(vs.nz)[np.newaxis, np.newaxis, :] == (vs.kbot - 1)[:, :, np.newaxis]

    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
    dtr_speed = (vs.wd0 + vs.mw * np.where(-zw < vs.mwz, -zw, vs.mwz)) \
        * vs.maskT

    detritus = Recyclable_tracer(vs.detritus, 'detritus',
                                 sinking_speed=dtr_speed,
                                 recycling_rate=vs.remineralization_rate_detritus)

    phytoplankton = Phytoplankton(vs.phytoplankton, 'phytoplankton',
                                  light_attenuation=vs.light_attenuation_phytoplankton,
                                  growth_parameter=vs.maximum_growth_rate_phyto,
                                  recycling_rate=vs.fast_recycling_rate_phytoplankton,
                                  mortality_rate=vs.specific_mortality_phytoplankton)

    zooplankton = Zooplankton(vs.zooplankton, 'zooplankton',
                              max_grazing=vs.maximum_grazing_rate,
                              grazing_saturation_constant=vs.saturation_constant_Z_grazing,
                              assimilation_efficiency=vs.assimilation_efficiency,
                              growth_efficiency=vs.zooplankton_growth_efficiency,
                              grazing_preferences=vs.zprefs,
                              mortality_rate=vs.quadric_mortality_zooplankton)

    po4 = NPZD_tracer(vs.po4, 'po4')

    # Add tracers to the model
    register_npzd_data(vs, detritus)
    register_npzd_data(vs, phytoplankton)
    register_npzd_data(vs, zooplankton)
    register_npzd_data(vs, po4)

    vs.limiting_functions['phytoplankton'] = [phosphate_limitation_phytoplankton]

    # Zooplankton preferences for grazing on keys
    # Values are scaled automatically at the end of this function
    vs.zprefs['phytoplankton'] = vs.zprefP
    vs.zprefs['zooplankton'] = vs.zprefZ
    vs.zprefs['detritus'] = vs.zprefDet

    # Register rules for interactions between active tracers
    register_npzd_rule(vs, 'npzd_basic_phytoplankton_grazing',
                       (grazing, 'phytoplankton', 'zooplankton'),
                       label='Grazing')
    register_npzd_rule(vs, 'npzd_basic_phytoplankton_mortality',
                       (mortality, 'phytoplankton', 'detritus'),
                       label='Mortality')
    register_npzd_rule(vs, 'npzd_basic_phytoplankton_fast_recycling',
                       (recycling_to_po4, 'phytoplankton', 'po4'),
                       label='Fast recycling')
    register_npzd_rule(vs, 'npzd_basic_zooplankton_grazing',
                       (empty_rule, 'zooplankton', 'zooplankton'),
                       label='Grazing')
    register_npzd_rule(vs, 'npzd_basic_zooplankton_mortality',
                       (mortality, 'zooplankton', 'detritus'),
                       label='Mortality')
    register_npzd_rule(vs, 'npzd_basic_zooplankton_sloppy_feeding',
                       (sloppy_feeding, 'zooplankton', 'detritus'),
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
        'npzd_basic_phytoplankton_fast_recycling',
        'npzd_basic_phytoplankton_primary_production',
        'npzd_basic_zooplankton_grazing',
        'npzd_basic_zooplankton_mortality',
        'npzd_basic_zooplankton_sloppy_feeding',
        'npzd_basic_detritus_remineralization',
        'npzd_basic_detritus_grazing',
        'npzd_basic_detritus_bottom_remineralization'
    ])

    register_npzd_rule(vs, 'empty_rule', (empty_rule, None, None))


@veros_method(inline=True)
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

    from .npzd_tracers import NPZD_tracer

    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird

    # redistribution fraction for calcite at level k
    vs.rcak[:, :, :-1] = (- np.exp(zw[:-1] / vs.dcaco3) + np.exp(zw[1:] / vs.dcaco3)) / vs.dzt[:-1]
    vs.rcak[:, :, -1] = - (np.exp(zw[-1] / vs.dcaco3) - 1.0) / vs.dzt[-1]

    # redistribution fraction at bottom
    rcab = np.empty_like(vs.dic[..., 0])
    rcab[:, : -1] = 1 / vs.dzt[-1]
    rcab[:, :, :-1] = np.exp(zw[:-1] / vs.dcaco3) / vs.dzt[1:]

    # merge bottom into level k and reset every cell outside ocean
    vs.rcak[vs.bottom_mask] = rcab[vs.bottom_mask]
    vs.rcak[...] *= vs.maskT

    # Need to track dissolved inorganic carbon, alkalinity
    dic = NPZD_tracer(vs.dic, 'DIC')
    alk = NPZD_tracer(vs.alkalinity, 'alkalinity')
    register_npzd_data(vs, dic)
    register_npzd_data(vs, alk)

    # Only for collection purposes - to be redistributed in post rules
    caco3 = NPZD_tracer(np.zeros_like(vs.dic), "caco3", transport=False)
    register_npzd_data(vs, caco3)

    # Atmosphere
    register_npzd_rule(vs, 'npzd_carbon_flux',
                       (co2_surface_flux, 'co2', 'DIC'),
                       boundary='SURFACE', group='PRE',
                       label='Atmosphere exchange')

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
                       group='POST',
                       label='Changes in DIC reflected in ALK')
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


@veros_method(inline=True)
def setupNPZD(vs):
    """Taking veros variables and packaging them up into iterables"""

    setup_basic_npzd_rules(vs)

    # Add carbon to the model
    if vs.enable_carbon:
        setup_carbon_npzd_rules(vs)

    from .npzd_rules import excretion
    register_npzd_common_source_rule(vs, 'npzd_basic_zooplankton_excretion',
                                     (excretion, 'zooplankton', 'po4'),
                                     label='Excretion')

    register_npzd_common_source_rule(vs, 'npzd_basic_zooplankton_excretion',
                                     (excretion, 'zooplankton', 'DIC'),
                                     label='Excretion')

    # Turn common source rules into selectable rules
    # We should not have to make this check, it should just be defined
    if hasattr(vs, 'common_source_rules'):
        # All common source rules have been saved to a dictionary
        # where the key is the collection identifier and individual rules are
        # named after the convention {collection_name}_{tracer_name}
        for name, rules in vs.common_source_rules.items():
            collection = [name + "_" + rules[0][1]] + [name + "_" + rules[i][2] for i in range(1, len(rules))]
            register_npzd_rule(vs, name, collection)

    for rule in vs.npzd_selected_rules:
        select_npzd_rule(vs, rule)

    # Update Zooplankton preferences dynamically
    # Ideally this would be done in the Zooplankton class
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
    r"""
    Main driving function for NPZD functionality

    Computes transport terms and biological activity separately

    \begin{equation}
        \dfrac{\partial C_i}{\partial t} = T + S
    \end{equation}
    """

    # TODO: Refactor transportation code to be defined only once and also used by thermodynamics
    # TODO: Dissipation on W-grid if necessary

    npzd_changes = biogeochemistry(vs)

    """
    For vertical mixing
    """

    a_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    b_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    c_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    d_tri = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)
    delta = allocate(vs, ('xt', 'yt', 'zt'), include_ghosts=False)

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

    for tracer in vs.npzd_tracers.values():
        utilities.enforce_boundaries(vs, tracer)
