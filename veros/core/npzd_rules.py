"""
Collection of rules to be used by the npzd module
"""
from .. import veros_method
from . import atmospherefluxes


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
    """ Zooplankton grows by amount digested, eaten decreases by amount grazed """
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
    # return {nutrient: vs.redfield_ratio_PN * vs.excretion_total}

@veros_method
def primary_production_from_DIC(vs, nutrient, plankton):
    """ Only using DIC, because plankton is handled by po4 ... shitty design right now """
    return {nutrient: - vs.redfield_ratio_CN * vs.net_primary_production[plankton]}

@veros_method
def recycling_to_po4(vs, plankton, phosphate):
    """ Recycling to phosphate is scaled by redfield ratio P to N """
    return recycling(vs, plankton, phosphate, vs.redfield_ratio_PN)

@veros_method
def recycling_to_no3(vs, plankton, no3):
    """ Recycling to nitrate needs no scaling """
    return {no3: recycling(vs, plankton, no3, 1)[no3]}

@veros_method
def recycling_to_dic(vs, plankton, dic):
    """ Recycling to carbon is scaled by redfield ratio C to N removing plankton has already been done by po4"""
    return {dic: recycling(vs, plankton, dic, vs.redfield_ratio_CN)[dic]}

@veros_method
def recycling_phyto_to_dic(vs, plankton, dic):
    """ Fast recycling of phytoplankton to DIC """
    return {dic: recycling(vs, plankton, dic, (1 - vs.dfrt) * vs.redfield_ratio_CN)[dic]}

@veros_method
def excretion_dic(vs, zooplankton, nutrient):
    """ Zooplankton excretes nutrients after eating. Poop, breathing... """
    return {nutrient: vs.redfield_ratio_CN * vs.excretion_total}


@veros_method
def calcite_production(vs, plankton, DIC, calcite):
    """ Calcite is produced at a rate similar to detritus
        Intended for use with a smoothing rule
        If explicit tracking of calcite is desired use
        rules for the explicit relationship
    """

    # changes to production of calcite
    dprca = (vs.mortality[plankton] + vs.grazing[plankton] *
             (1 - vs.assimilation_efficiency)) * vs.capr * vs.redfield_ratio_CN * 1e-3

    return {DIC: -dprca, calcite: dprca}


@veros_method
def calcite_production_alk(vs, plankton, alkalinity, calcite):
    """ alkalinity is scaled by DIC consumption """
    return {alkalinity: 2 * calcite_production(vs, plankton, "DIC", calcite)["DIC"]}


@veros_method
def calcite_production_phyto(vs, DIC, calcite):
    """ DIC is consumed to produce calcite. How much depends on the mortality
        and sloppy feeding of and on phytoplankton and zooplankton
    """
    return calcite_production(vs, "phytoplankton", DIC, calcite)


@veros_method
def calcite_production_phyto_alk(vs, alkalinity, calcite):
    """ sclaed version off calcite_production_phyto for alkalinity """
    return calcite_production_alk(vs, "phytoplankton", alkalinity, calcite)


@veros_method
def post_redistribute_calcite(vs, calcite, tracer):
    """ Post rule to redistribute produced calcite """
    total_production = (vs.temporary_tracers[calcite] * vs.dzt).sum(axis=2)
    redistributed_production = total_production[:, :, np.newaxis] * vs.rcak
    return {tracer: redistributed_production}


@veros_method
def pre_reset_calcite(vs, tracer, calcite):
    """ Pre rule to reset calcite production """
    return {calcite: - vs.temporary_tracers[calcite]}


# @veros_method
# def recycling_to_alk(vs, detritus, alkalinity):
#     """ This should be turned into a combined rule with DIC, as they just have opposite terms, but only withdraw from detritus once """
#     return {alkalinity: - recycling_to_dic(vs, detritus, "DIC")["DIC"] / vs.redfield_ratio_CN * 1e-3}


# @veros_method
# def primary_production_from_alk(vs, alkalinity, plankton):
#     """ Single entry, should be merged with DIC """
#     return {alkalinity: - primary_production_from_DIC(vs, "DIC", plankton)["DIC"] / vs.redfield_ratio_CN * 1e-3}


# @veros_method
# def recycling_phyto_to_alk(vs, plankton, alkalinity):
#     """ Single entry should be merged with DIC """
#     return {alkalinity: - recycling_phyto_to_dic(vs, plankton, "DIC")["DIC"] / vs.redfield_ratio_CN * 1e-3}


# @veros_method
# def excretion_alk(vs, plankton, alkalinity):
#     """ Single entry, should be merged with DIC """
#     return {alkalinity: -excretion_dic(vs, plankton, "DIC")["DIC"] / vs.redfield_ratio_CN * 1e-3}


@veros_method
def co2_surface_flux(vs, co2, dic):
    """ Pre rule to add or remove DIC from surface layer """
    atmospherefluxes.carbon_flux(vs)
#    vs.cflux[:, :] = atmospherefluxes.carbon_flux(vs)  # TODO move this somewhere else
    # flux = np.zeros((vs.cflux.shape[0], vs.cflux.shape[1], vs.nz))
    # flux[:, :, -1] = vs.cflux * vs.dt_tracer / vs.dzt[-1]
    flux = vs.cflux * vs.dt_tracer / vs.dzt[-1]
    return {dic: flux}  # NOTE we don't have an atmosphere, so this rules is a stub


# @veros_method
# def co2_surface_flux_alk(vs, co2, alk):
#     """
#     Pre rule to add or remove alkalinity from surface layer
#     assumes co2_surface_flux has been run for the current timestep
#     """
#     flux = np.zeros((vs.cflux.shape[0], vs.cflux.shape[1], vs.nz))
#     flux[:, :, -1] = - vs.cflux * vs.dt_tracer / vs.dzt[-1] / vs.redfield_ratio_CN
#     return {alk: flux}

@veros_method
def primary_production_from_dop_po4(vs, DOP, plankton):
    """
    Plankton growth by DOP uptake.
    """

    return {DOP: - vs.net_primary_production[DOP] * vs.dop_consumption, plankton: vs.net_primary_production[plankton] * vs.dop_consumption}


@veros_method
def primary_production_from_po4_dop(vs, po4, plankton):
    """
    Plankton growth by po4 consumption
    """

    cons = np.logical_not(vs.dop_consumption)
    return {po4: - vs.net_primary_production[plankton] * cons, plankton: vs.net_primary_production[plankton] * cons}

@veros_method
def diazotroph_growth_don(vs, DON, diazotroph):
    """
    Diazotroph growth by DON consumption
    """

    return {}

@veros_method
def dic_alk_scale(vs, DIC, alkalinity):
    """ Redistribute change in DIC as change in alkalinity """
    return {alkalinity: (vs.temporary_tracers[DIC] - vs.npzd_tracers[DIC][:, :, :, vs.tau]) / vs.redfield_ratio_CN}# * 1e-3}
