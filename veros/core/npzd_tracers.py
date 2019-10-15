"""
Classes for npzd tracers
"""
import numpy as np
from .. import veros_method

class NPZD_tracer(np.ndarray):
    """
    Class for npzd tracers to store additional information about themselves.
    Inhenrits from numpy.ndarray to make it work seamless with array operations
    """

    # These are not used. They are just here for documenting the parameters set in __new__
    name = None  #: A unique identifier (during the configured simulation)
    sinking_speed = None  #: Numpy array for how fast the tracer sinks in each cell
    input_array = None  #: Numpy array backing data
    description = "This is the base tracer class"  #: Metadata
    transport = True  #: Whether or not to include the tracer in physical transport
    light_attenuation = None  #: Factor for how much light is blocked


    def __new__(cls, input_array, name, sinking_speed=None, light_attenuation=None, transport=True,
                description = None):
        obj = np.asarray(input_array).view(cls)
        if sinking_speed is not None:
            obj.sinking_speed = sinking_speed
        if light_attenuation is not None:
            obj.light_attenuation = light_attenuation


        obj.name = name
        obj.transport = transport


        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        # If we are slicing, obj will have __dir__ therefore we need to set attributes
        # on new sliced array
        if hasattr(obj, "__dir__"):
            for attribute in (set(dir(obj)) - set(dir(self))):
                setattr(self, attribute, getattr(obj, attribute))



class Recyclable_tracer(NPZD_tracer):
    """
    A recyclable tracer, this would be tracer, which may be a tracer like detritus, which can be recycled
    """

    # this is just for documentation
    recycling_rate = 0  #: A factor scaling the recycling by the population size

    def __new__(cls, input_array, name, recycling_rate=0, **kwargs):
        obj = super().__new__(cls, input_array, name, **kwargs)
        obj.recycling_rate = recycling_rate

        return obj

    @veros_method(inline=True)
    def recycle(self, vs):
        """
        Recycling is temperature dependant by vs.bct
        """
        return vs.bct * self.recycling_rate * self



class Plankton(Recyclable_tracer):
    """
    Class for plankton object, which is both recyclable and displays mortality
    Typically, it would desirable to also set light attenuation
    This class is intended as a base for phytoplankton and zooplankton and not
    as a standalone class

    """

    # this is just for documentation
    mortality_rate = 0  #: Rate at which the tracer is dying in mortality method

    def __new__(cls, input_array, name, mortality_rate=0, **kwargs):
        obj = super().__new__(cls, input_array, name, **kwargs)
        obj.mortality_rate = mortality_rate


        return obj

    @veros_method(inline=True)
    def mortality(self, vs):
        """
        The mortality rate scales with population size
        """
        return self * self.mortality_rate


class Phytoplankton(Plankton):
    """
    Phytoplankton also has primary production
    """

    # this is just for documentation
    growth_parameter = 0  #: Scaling factor for maximum potential growth

    def __new__(cls, input_array, name, growth_parameter=0, **kwargs):
        obj = super().__new__(cls, input_array, name, **kwargs)
        obj.growth_parameter = growth_parameter

        return obj


    @veros_method(inline=True)
    def potential_growth(self, vs, grid_light, light_attenuation):
        """ Light limited growth, not limited growth """
        f1 = np.exp(-light_attenuation)  # available light
        jmax = self.growth_parameter * vs.bct  # maximum growth
        gd = jmax * vs.dayfrac[np.newaxis, :, np.newaxis]  # growth in fraction of day
        avej = self._avg_J(vs, f1, gd, grid_light, light_attenuation)  # light limited growth

        return jmax, avej


    @veros_method(inline=True)
    def _avg_J(self, vs, f1, gd, grid_light, light_attenuation):
        """Average J"""
        u1 = np.maximum(grid_light / gd, vs.u1_min)
        u2 = u1 * f1

        # NOTE: There is an approximation here: u1 < 20
        phi1 = np.log(u1 + np.sqrt(1 + u1**2)) - (np.sqrt(1 + u1**2) - 1) / u1
        phi2 = np.log(u2 + np.sqrt(1 + u2**2)) - (np.sqrt(1 + u2**2) - 1) / u2

        return gd * (phi1 - phi2) / light_attenuation


class Zooplankton(Plankton):
    """
    Zooplankton displays quadratic mortality rate but otherwise is similar to ordinary phytoplankton
    """
    max_grazing = 0  #: Scaling factor for maximum grazing rate
    grazing_saturation_constant = 1  #: Saturation in Michaelis-Menten
    grazing_preferences = {}  #: Dictionary of preferences for grazing on other tracers
    assimilation_efficiency = 0  #: Fraction of grazed material ingested
    growth_efficiency = 0  #: Fraction of ingested material resulting in growth

    def __new__(cls, input_array, name, max_grazing=0, grazing_saturation_constant=1,
                grazing_preferences={}, assimilation_efficiency=0,
                growth_efficiency=0, **kwargs):
        obj = super().__new__(cls, input_array, name, **kwargs)
        obj.max_grazing = max_grazing  #: Scaling factor for maximum grazing rate
        obj.grazing_saturation_constant = grazing_saturation_constant  #: Saturation in Michaelis-Menten
        obj.grazing_preferences = grazing_preferences  #: Dictionary of preferences for grazing on other tracers
        obj.assimilation_efficiency = assimilation_efficiency  #: Fraction of grazed material ingested
        obj.growth_efficiency = growth_efficiency  #: Fraction of ingested material resulting in growth
        obj._gmax = 0  # should be private

        return obj

    @veros_method(inline=True)
    def update_internal(self, vs):
        """
        Updates internal numbers, which are calculated only from Veros values
        """
        self._gmax = self.max_grazing * vs.bbio ** (vs.cbio * np.minimum(20, vs.temp[..., vs.tau]))


    @veros_method(inline=True)
    def mortality(self, vs):
        """
        Zooplankton is modelled with a quadratic mortality
        """
        return self.mortality_rate * self ** 2


    @veros_method(inline=True)
    def grazing(self, vs, tracers, flags):
        """
        Zooplankton grazing returns total grazing, digestion i.e. how much is available
        for zooplankton growth, excretion and sloppy feeding
        All are useful to have calculated once and made available to rules
        """

        # TODO check saturation constants
        thetaZ = sum([pref_score * tracers[preference] for preference, pref_score
                      in self.grazing_preferences.items()])\
                      + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

        ingestion = {preference: pref_score / thetaZ for preference, pref_score in self.grazing_preferences.items()}

        grazing = {preference: flags[preference] * flags[self.name] * self._gmax *
                   ingestion[preference] * tracers[preference] * self
                   for preference in ingestion}


        digestion = {preference: self.assimilation_efficiency * amount_grazed
                     for preference, amount_grazed in grazing.items()}

        excretion = {preference: (1 - self.growth_efficiency) * amount_digested
                     for preference, amount_digested in digestion.items()}

        sloppy_feeding = {preference: grazing[preference] - digestion[preference]
                          for preference in grazing}

        return grazing, digestion, excretion, sloppy_feeding
