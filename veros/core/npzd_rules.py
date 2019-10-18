"""
Collection of rules to be used by the npzd module

Rules should always take 3 arguments:
    1. vs
    2. Name of source tracer
    3. Name of sink tracer
"""
from .. import veros_method
from . import atmospherefluxes


@veros_method(inline=True)
def empty_rule(*args):
    """An empty rule for providing structure"""
    return {}


@veros_method(inline=True)
def primary_production(vs, nutrient, plankton):
    """Primary production: Growth by consumption of light and nutrients"""
    return {nutrient: - vs.redfield_ratio_PN * vs.net_primary_production[plankton], plankton: vs.net_primary_production[plankton]}


@veros_method(inline=True)
def recycling(vs, plankton, nutrient, ratio):
    """Plankton or detritus is recycled into nutrients

    Parameters
    ----------
    ratio
        Factor for adjusting nutrient contribution. Typically a Redfield ratio.
    """
    return {plankton: - vs.recycled[plankton], nutrient: ratio * vs.recycled[plankton]}


@veros_method(inline=True)
def mortality(vs, plankton, detritus):
    """All dead matter from plankton is converted to detritus"""
    return {plankton: - vs.mortality[plankton], detritus: vs.mortality[plankton]}


@veros_method(inline=True)
def sloppy_feeding(vs, zooplankton, detritus):
    """When zooplankton graces, some is not eaten. This is converted to detritus."""
    sloppy_sum = sum(vs.sloppy_feeding.values())
    return {zooplankton: -sloppy_sum, detritus: sloppy_sum}


@veros_method(inline=True)
def grazing(vs, eaten, zooplankton):
    """Zooplankton grows by amount digested, eaten decreases by amount grazed"""
    return {eaten: - vs.grazing[eaten], zooplankton: vs.grazing[eaten]}


@veros_method(inline=True)
def excretion(vs, zooplankton, nutrient):
    """Zooplankton excretes nutrients after eating. Fecal matter, breathing..."""
    return {zooplankton: - vs.excretion_total, nutrient: vs.redfield_ratio_PN * vs.excretion_total}


@veros_method(inline=True)
def primary_production_from_DIC(vs, nutrient, plankton):
    """Stub rule for primary production consuming DIC. Should be replaced by common sink rule"""
    return {nutrient: - vs.redfield_ratio_CN * vs.net_primary_production[plankton]}


@veros_method(inline=True)
def recycling_to_po4(vs, plankton, phosphate):
    """Recycling to phosphate is scaled by redfield ratio P to N"""
    return recycling(vs, plankton, phosphate, vs.redfield_ratio_PN)


@veros_method(inline=True)
def recycling_to_no3(vs, plankton, no3):
    """Recycling to nitrate needs no scaling"""
    return {no3: recycling(vs, plankton, no3, 1)[no3]}


@veros_method(inline=True)
def recycling_to_dic(vs, plankton, dic):
    """Recycling to carbon

    Is scaled by redfield ratio C to N removing plankton has already been done by po4
    """
    return {dic: recycling(vs, plankton, dic, vs.redfield_ratio_CN)[dic]}


@veros_method(inline=True)
def recycling_phyto_to_dic(vs, plankton, dic):
    """Fast recycling of phytoplankton to DIC"""
    return {dic: recycling(vs, plankton, dic, vs.redfield_ratio_CN)[dic]}


@veros_method(inline=True)
def excretion_dic(vs, zooplankton, nutrient):
    """Zooplankton excretes nutrients after eating. Poop, breathing..."""
    return {nutrient: vs.redfield_ratio_CN * vs.excretion_total}


@veros_method(inline=True)
def calcite_production(vs, plankton, DIC, calcite):
    """Calcite is produced at a rate similar to detritus

    Intended for use with a smoothing rule
    If explicit tracking of calcite is desired use
    rules for the explicit relationship
    """

    # changes to production of calcite
    dprca = (vs.mortality[plankton] + vs.grazing[plankton] *
             (1 - vs.assimilation_efficiency)) * vs.capr * vs.redfield_ratio_CN

    return {DIC: -dprca, calcite: dprca}


@veros_method(inline=True)
def calcite_production_alk(vs, plankton, alkalinity, calcite):
    """alkalinity is scaled by DIC consumption"""
    return {alkalinity: 2 * calcite_production(vs, plankton, 'DIC', calcite)['DIC']}


@veros_method(inline=True)
def calcite_production_phyto(vs, DIC, calcite):
    """DIC is consumed to produce calcite.

    How much depends on the mortality and sloppy feeding of and on phytoplankton and zooplankton
    """
    return calcite_production(vs, 'phytoplankton', DIC, calcite)


@veros_method(inline=True)
def calcite_production_phyto_alk(vs, alkalinity, calcite):
    """Scaled version off calcite_production_phyto for alkalinity"""
    return calcite_production_alk(vs, 'phytoplankton', alkalinity, calcite)


@veros_method(inline=True)
def post_redistribute_calcite(vs, calcite, tracer):
    """Post rule to redistribute produced calcite"""
    total_production = (vs.temporary_tracers[calcite] * vs.dzt).sum(axis=2)
    redistributed_production = total_production[:, :, np.newaxis] * vs.rcak
    return {tracer: redistributed_production}


@veros_method(inline=True)
def post_redistribute_calcite_alk(vs, calcite, alkalinity):
    """Post rule reusing post_redistribute_calcite to redistribute alkalinity"""
    return {alkalinity: 2 * post_redistribute_calcite(vs, calcite, 'DIC')['DIC']}


@veros_method(inline=True)
def pre_reset_calcite(vs, tracer, calcite):
    """Pre rule to reset calcite production"""
    return {calcite: - vs.temporary_tracers[calcite]}


@veros_method
def co2_surface_flux(vs, co2, dic):
    """Pre rule to add or remove DIC from surface layer"""
    atmospherefluxes.carbon_flux(vs)
    flux = vs.cflux * vs.dt_tracer / vs.dzt[-1]
    return {dic: flux}  # NOTE we don't have an atmosphere, so this rules is a stub


@veros_method(inline=True)
def dic_alk_scale(vs, DIC, alkalinity):
    """Redistribute change in DIC as change in alkalinity"""
    return {alkalinity: (vs.temporary_tracers[DIC] - vs.npzd_tracers[DIC][:, :, :, vs.tau]) / vs.redfield_ratio_CN}


@veros_method(inline=True)
def bottom_remineralization(vs, source, sink, scale):
    """Exported material falling through the ocean floor is converted to nutrients

    Note
    ----
    There can be no source, because that is handled by the sinking code

    Parameters
    ----------
    scale
        Factor to convert remineralized material to nutrient. Typically Redfield ratio.
    """
    return {sink: vs.npzd_export[source][vs.bottom_mask] * scale}


@veros_method(inline=True)
def bottom_remineralization_detritus_po4(vs, detritus, po4):
    """Remineralize detritus at the bottom to PO4"""
    return bottom_remineralization(vs, detritus, po4, vs.redfield_ratio_PN)


@veros_method(inline=True)
def bottom_remineralization_detritus_DIC(vs, detritus, DIC):
    """Remineralize detritus at the bottom to DIC"""
    return bottom_remineralization(vs, detritus, DIC, vs.redfield_ratio_CN)
