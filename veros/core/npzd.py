"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
import numpy as np  # NOTE np is already defined somehow
from graphviz import Digraph
from .. import veros_method
from . import advection, diffusion, thermodynamics, cyclic


@veros_method
def biogeochemistry(vs):
    """
    Integrate biochemistry: phytoplankton, zooplankton, detritus, po4
    """

    # Number of timesteps to do for bio tracers
    nbio = int(vs.dt_mom // vs.dt_bio)

    # Integrated phytplankton
    phyto_integrated = np.zeros_like(vs.phytoplankton[:, :, 0])

    # import from layer above, export to layer below
    # impo = {"detritus": np.zeros((vs.detritus.shape[:2]))}
    # export = {"detritus": np.zeros((vs.detritus.shape[:2]))}
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

    for k in reversed(range(vs.nz)):

        tracers = {name: value[:, :, k] for name, value in vs.npzd_tracers.items()}

        # flags to prevent more outgoing flux than available capacity - stability?
        # flags = {tracer: True for tracer in tracers}
        flags = {tracer: True for tracer in tracers}

        # Set flags and tracers based on minimum concentration
        update_flags_and_tracers(vs, flags, tracers, refresh=True)

        # incomming radiation at layer
        swr = swr * np.exp(- vs.light_attenuation_phytoplankton * phyto_integrated)


        # TODO remove if statements in running code
        if vs.enable_calcifiers:
            swr[:, :] *= np.exp(- vs.light_attenuation_caco3 * tracers["caco3"])
            dissk1 = vs.dissk0 * (1 - vs.Omega_c)

        # integrated phytoplankton for use in next layer
        phyto_integrated = sum([np.maximum(tracers[plankton], vs.trcmin) for plankton in vs.plankton_types]) * vs.dzt[k]

        # light at top of gridbox
        grid_light = swr * np.exp(vs.zw[k] * vs.rctheta)

        # calculate import of sinking particles from layer above
        for sinker in vs.sinking_speeds:
            impo[sinker] = export[sinker] / (vs.dzt[k] * vs.dt_bio)
            export[sinker][:, :] = 0

        jmax, avej = phytoplankton_potential_growth(vs, k, grid_light, tracers, bct)

        for _ in range(nbio):
            vs.net_primary_production = calc_net_primary_production(vs, tracers, flags, jmax["phytoplankton"], avej["phytoplankton"], vs.saturation_constant_N)

            vs.recycled = {"phytoplankton": flags["phytoplankton"] * vs.nupt0 * bct[:, :, k] * tracers["phytoplankton"],
                        "detritus": flags["detritus"] * vs.nud0 * bct[:, :, k] * tracers["detritus"]}

            vs.mortality = {"phytoplankton": flags["phytoplankton"] * vs.specific_mortality_phytoplankton * tracers["phytoplankton"],
                         "zooplankton": flags["zooplankton"] * vs.quadric_mortality_zooplankton * tracers["zooplankton"] * tracers["zooplankton"]}

            vs.grazing, vs.digestion, vs.excretion, vs.sloppy_feeding = zooplankton_grazing(vs, tracers, flags, gmax[:, :, k])
            vs.excretion_total = sum(vs.excretion.values())

            tmp_expo = {sinker: speed[:, :, k] * tracers[sinker] * flags[sinker] for sinker, speed in vs.sinking_speeds.items()}


            for name, value in tracers.items():
                vs.npzd_tracers[name][:, :, k] = value


            # npzd_updates = [rule[0](vs, vs.npzd_tracers[rule[1]], vs.npzd_tracers[rule[2]]) for rule in vs.npzd_rules]
            npzd_updates = [rule[0](vs, rule[1], rule[2]) for rule in vs.npzd_rules]
            for update in npzd_updates:
                for key, value in update.items():
                    vs.npzd_tracers[key][:, :, k] += value * vs.dt_bio

            # vs.npzd_tracers["detritus"][:, :, k] += impo["detritus"] - tmp_expo["detritus"]
            for tracer in vs.sinking_speeds:
                vs.npzd_tracers[tracer][:, :, k] += impo[tracer] - tmp_expo[tracer]

            for name in vs.npzd_tracers:
                tracers[name] = vs.npzd_tracers[name][:, :, k]

            for sinker in vs.sinking_speeds:
                export[sinker][:, :] += tmp_expo[sinker] * vs.dt_bio

            update_flags_and_tracers(vs, flags, tracers)

        for sinker in export:
            export[sinker][:, :] *= vs.dzt[k] / nbio

        for name, value in tracers.items():
            vs.npzd_tracers[name][:, :, k] = value


@veros_method
def zooplankton_grazing(vs, tracers, flags, gmax):
    zprefs = {"phytoplankton": vs.zprefP, "zooplankton": vs.zprefZ, "detritus": vs.zprefDet}

    thetaZ = sum([pref_score * tracers[preference] for preference, pref_score in zprefs.items()]) + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

    ingestion = {preference: pref_score / thetaZ for preference, pref_score in zprefs.items()}

    grazing = {preference: flags[preference] * flags["zooplankton"] * gmax * ingestion[preference] * tracers[preference] * tracers["zooplankton"] for preference in ingestion}

    digestion = {preference: vs.assimilation_efficiency * amount_grazed for preference, amount_grazed in grazing.items()}

    excretion = {preference: vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

    sloppy_feeding = {preference: (1 - vs.assimilation_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}

    return grazing, digestion, excretion, sloppy_feeding



@veros_method
def phytoplankton_potential_growth(vs, k, grid_light, tracers, bct):
    jmax = {"phytoplankton": 0}
    gd = {"phytoplankton": 0}
    avej = {"phytoplankton": 0}
    gl = {"phytoplankton": 0}

    light_attenuation = vs.dzt[k] * (vs.light_attenuation_water + vs.light_attenuation_phytoplankton * tracers["phytoplankton"])

    f1 = np.exp(-light_attenuation)
    gl["phytoplankton"] = grid_light
    jmax["phytoplankton"] = vs.abio_P * bct[:, :, k]
    gd["phytoplankton"] = jmax["phytoplankton"] * vs.dt_mom
    avej["phytoplankton"] = avg_J(vs, f1, gd["phytoplankton"], gl["phytoplankton"], light_attenuation)

    return jmax, avej

@veros_method
def calc_net_primary_production(vs, tracers, flags, jmax, avej, saturation_constant):
    limit_phosphate = tracers["po4"] / (saturation_constant + tracers["po4"])
    u = limit_phosphate * jmax
    u = np.minimum(u, avej)
    net_primary_production = u * tracers["phytoplankton"] * flags["po4"]

    return net_primary_production



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
def register_npzd_data(vs, name, value=0):
    vs.npzd_tracers[name] = value
    vs.npzd_graph.attr('node', shape='square')
    vs.npzd_graph.node(name)

@veros_method
def register_npzd_rule(vs, function, source, destination, label="?"):
    vs.npzd_rules.append((function, source, destination))
    vs.npzd_graph.edge(source, destination, label=label)

@veros_method
def empty_rule(*args):
    return {}

@veros_method
def primary_production(vs, nutrient, plankton):
    return {nutrient: - vs.redfield_ratio_PN * vs.net_primary_production, plankton: vs.net_primary_production}

@veros_method
def recycling(vs, plankton, nutrient):
    return {nutrient: vs.redfield_ratio_PN * vs.recycled[plankton], plankton: - vs.recycled[plankton]}

@veros_method
def mortality(vs, plankton, detritus):
    return {plankton: - vs.mortality[plankton], detritus: vs.mortality[plankton]}

@veros_method
def sloppy_feeding(vs, plankton, detritus):
    return {detritus: vs.sloppy_feeding[plankton]}

@veros_method
def grazing(vs, eaten, zooplankton):
    return {eaten: - vs.grazing[eaten], zooplankton: vs.digestion[eaten]}

@veros_method
def excretion(vs, zooplankton, nutrient):
    return {zooplankton: - vs.excretion_total, nutrient: vs.redfield_ratio_PN * vs.excretion_total}

@veros_method
def calpro_zooplankton(vs, tracer, caco3):
    """tracer is generic term. Should be used for dic and alcalinity"""

    cp = (vs.sloppy_feeding["zooplankton"] + vs.mortality["zooplankton"]) * vs.capr * vs.redfield_ratio_CP * vs.redfield_ratio_PN

    return {caco3: cp, tracer: -cp}

@veros_method
def primary_production_from_DIC(vs, nutrient, plankton):
    return {nutrient: - vs.redfield_ratio_CN * vs.net_primary_production, plankton: 0}
    # ca = (vs.net_primary_production["coccolitophore"] - vs.mortality["coccolitophore"] - vs.recycled["coccolitophore"] - vs.grazing["coccolitophore"] + digestion_total - vs.mortality["zooplankton"] - vs.grazing["zooplankton"] - vs.excretion_total) * vs.capr * vs.redfield_ratio_CP * vs.redfield_ratio_PN * 1e3

@veros_method
def recycling_DIC(vs, detritus, nutrient):
    return {nutrient: vs.redfield_ratio_CN * vs.recycled[detritus], detritus: - vs.recycled[detritus]}


@veros_method
def recycling_DIC_phyto(vs, plankton, nutrient):
    return {nutrient: (1 - vs.dfrt) * vs.redfield_ratio_CN * vs.recycled[plankton], plankton: - vs.dfrt * vs.recycled[plankton]}



            #     dtracer["caco3"][:, :] += calpro[:, :, k] - dissl[:, :, k]

#             tracers["caco3"][:, :] += vs.dt_bio * (calpro[:, :, k] - dissl[:, :, k] - tmp_expo["caco3"] + impo["caco3"])

        # for k in range(vs.nz - 1):
        #     vs.npzd_tracers["DIC"][:, :, k] += dissl[:, :, k] / nbio * 1e-3 - calpro[:, :, k] / nbio - calatt[:, :, k] * 1e-3
        #     vs.npzd_tracers["alkalinity"][:, :, k] += 2 * dissl[:, :, k] / nbio * 1e-3 - 2 * calpro[:, :, k] / nbio - 2 * calatt[:, :, k] * 1e-3

        # vs.npzd_tracers["DIC"][:, :, -1] += dissl[:, :, -1] / nbio * 1e-3 - calpro[:, :, -1] / nbio * 1e-3 - calatt[:, :, vs.nz - 1] / nbio * 1e-3 + export["caco3"] * 1e-3
        # vs.npzd_tracers["alkalinity"][:, :, -1] += 2 * dissl[:, :, -1] / nbio * 1e-3 - 2 * calpro[:, :, -1] / nbio * 1e-3 - 2 * calatt[:, :, -1] / nbio * 1e-3 + 2 * export["caco3"] * 1e-3




@veros_method
def setupNPZD(vs):
    """Taking veros variables and packaging them up into iterables"""
    vs.npzd_tracers = {}
    vs.npzd_rules = []
    vs.npzd_graph = Digraph("npzd_dynamics", filename="npzd_dynamics.gv")

    vs.sinking_speeds = {}
    vs.sinking_speeds["detritus"] = (vs.wd0 + vs.mw * np.where(-vs.zw < vs.mwz, -vs.zw, vs.mwz)) / vs.dzt * vs.maskT

    vs.plankton_types = ["phytoplankton"]

    register_npzd_data(vs, "phytoplankton", vs.phytoplankton)
    register_npzd_data(vs, "zooplankton", vs.zooplankton)
    register_npzd_data(vs, "detritus", vs.detritus)
    register_npzd_data(vs, "po4", vs.po4)

    register_npzd_rule(vs, grazing, "phytoplankton", "zooplankton", label="Grazing")
    register_npzd_rule(vs, mortality, "phytoplankton", "detritus", label="Mortality")
    register_npzd_rule(vs, sloppy_feeding, "phytoplankton", "detritus", label="Sloppy feeding")
    register_npzd_rule(vs, recycling, "phytoplankton", "po4", label="Fast recycling")
    register_npzd_rule(vs, grazing, "zooplankton", "zooplankton", label="Grazing")
    register_npzd_rule(vs, excretion, "zooplankton", "po4", label="Excretion")
    register_npzd_rule(vs, sloppy_feeding, "zooplankton", "detritus", label="Sloppy feeding")
    register_npzd_rule(vs, sloppy_feeding, "detritus", "detritus", label="Sloppy feeding")
    register_npzd_rule(vs, grazing, "detritus", "zooplankton", label="Grazing")
    register_npzd_rule(vs, recycling, "detritus", "po4", label="Remineralization")
    register_npzd_rule(vs, primary_production, "po4", "phytoplankton", label="Primary production")

    if vs.enable_nitrogen:
        pass

    if vs.enable_calcifiers:
        vs.redfield_ratio_CN = vs.redfield_ratio_CP * vs.redfield_ratio_PN

        vs.sinking_speeds["caco3"] = (vs.wc0 + vs.mw_c * np.where(-vs.zw < vs.mwz, -vs.zw, vs.mwz)) / vs.dzt * vs.maskT

        register_npzd_data(vs, "caco3", vs.caco3)
        register_npzd_data(vs, "DIC", vs.dic)
        register_npzd_data(vs, "alkalinity", vs.alkalinity)

        register_npzd_rule(vs, calpro_zooplankton, "caco3", "DIC", label="???????")
        register_npzd_rule(vs, calpro_zooplankton, "caco3", "alkalinity", label="???????")
        register_npzd_rule(vs, recycling_DIC, "detritus", "DIC", label="Remineralization")
        register_npzd_rule(vs, primary_production_from_DIC, "DIC", "phytoplankton", label="Primary production")
        register_npzd_rule(vs, recycling_DIC_phyto, "phytoplankton", "DIC", label="Fast recycling")
        register_npzd_rule(vs, excretion, "zooplankton", "DIC", label="Excretion")

    vs.npzd_graph_shown = False


    # vs.wd = np.empty_like(vs.detritus)
    # vs.wd[:, :] = (vs.wd0 + vs.mw * np.where(-vs.zw < vs.mwz, -vs.zw, vs.mwz)) / vs.dzt
    # vs.wd *= vs.maskT

@veros_method
def npzd(vs):
    """
    Main driving function for NPZD functionality
    Computes transport terms and biological activity separately

    $$
    \dfrac{\partial C_i}{\partial t} = T + S
    $$
    """
    if not vs.npzd_graph_shown and vs.show_npzd_graph:
        vs.npzd_graph.view()
        vs.npzd_graph_shown = True

    # commented out is T part
    vs.tracer_result = {key: 0 for key in vs.npzd_tracers}

    for tracer, val in vs.npzd_tracers.items():
        dNPZD_advect = np.zeros_like(val)
        dNPZD_diff = np.zeros_like(val)

        # NOTE Why is this in thermodynamics?
        thermodynamics.advect_tracer(vs, val, dNPZD_advect)
        diffusion.biharmonic(vs, val, 0.2, dNPZD_diff)  # TODO correct parameter
        vs.tracer_result[tracer] = vs.dt_mom * (dNPZD_advect + dNPZD_diff)

    biogeochemistry(vs)

    for tracer in vs.npzd_tracers:
        vs.npzd_tracers[tracer][...] += vs.tracer_result[tracer]

    if vs.enable_cyclic_x:
        for tracer in vs.npzd_tracers.values():
            cyclic.setcyclic_x(tracer)
