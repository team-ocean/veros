.. biogeochemistry:

Biogeochemistry
===============

The biogeochemistry module for Veros is designed for allowing construction of user defined ecosystems.
Base systems are made available for a basic Nutrients-Plankton-Zooplankton-Detritus, NPZD, system,
which can optionally be extended by a basic carbon cycle. Enabling the biogeochemistry module and
activating the basic NPZD system can be done by setting :attr:`enable_npzd = True`.

Ecosystems created with the biogeochemistry module are extensible and any components of them are
in principle replaceable. This is handled by three principles: Representation of model tracers by classes,
representation of interactions between tracers as rules, and separation of component creation from activation.


Tracer classes
--------------

All model tracers in the biogeochemistry module are created as instances of classes inheriting from a
base class :class:`NPZD_tracer`. This class itself inherits from numpy.ndarray, which allows using it
like any other Veros variable. The concentration (or appropriate unit) of the tracer within each cell
in Veros' grid is stored in the corresponding cell in the tracer grid. The base class defines attributes
for operations which may apply to any tracer.

Instances of this class must be created with a numpy array backing the tracer values. Preferably this
array was created in variables.py. Additionally a name must be supplied. This name will uniquely
identify the tracer during simulation. Optional arguments may be supplied:
:attr:`transport` By default this value is True. When :attr:`transport` is true, the tracer is transported
according to the selected transportation scheme. Setting a value for for :attr:`sinking_speed` will cause
the tracer to be included in calculations of sinking tracers. And setting :attr:`light_attenuation` will
block downward shortwave radiation proportionally to the concentration of the tracer.
This class may itself be used for tracers, which should not express any further features such
as the nutrients phosphate.

::

  NPZD_tracer(vs.po4, 'po4')

Tracers which should express actionable features such as grazing, primary production must implement
certain methods. Methods to implement are :attr:`mortality` for mortality, :attr:`recycle` for recycling,
:attr:`potential_growth` for primary production. In addition to this methods should be supplied a list
of functions representing limiting in growth by nutrients. :attr:`grazing` for grazing. This method
should return dictionaries for grazing, digestion, excretion, and sloppy feeding. Where the keys
are names of the tracers, which have been grazed upon.
It is possible to add additional actionable methods by editing npzd.py.

Predefined tracers
++++++++++++++++++

A number of classes for tracers have been predefined. These classes can be instantiated with different
parameters to defined tracers with varying properties. For example creating tracers for coccolithophores
and phytoplankton can be done like

::

  coccolithophores = Phytoplankton(np.zeros((3,vs.nx, vs.ny, vs.nz)), 'coccolithophores',
                                   light_attenuation=1,
                                   growth_parameter=0.9,
                                   recycling_rate=0.8,
                                   mortality_rate=0.7)



  phytoplankton = Phytoplankton(vs.phytoplankton, 'phytoplankton',
                                light_attenuation=vs.light_attenuation_phytoplankton,
                                growth_parameter=vs.maximum_growth_rate_phyto,
                                recycling_rate=vs.fast_recycling_rate_phytoplankton,
                                mortality_rate=vs.specific_mortality_phytoplankton)


Base tracer
###########

.. autoclass:: veros.core.npzd_tracers.NPZD_tracer

Recyclable tracer
#################

.. autoclass:: veros.core.npzd_tracers.Recyclable_tracer

Plankton
########

.. autoclass:: veros.core.npzd_tracers.Plankton


Phytoplankton
#############

.. autoclass:: veros.core.npzd_tracers.Phytoplankton


Zooplankton
###########

.. autoclass:: veros.core.npzd_tracers.Zooplankton


Extending tracers
+++++++++++++++++

The biogeochemistry tracers make use of the object oriented nature of Python to allow easy extensibility.
Tracers which exhibit nearly identical behavior can be created via extension. For example the
:class:`Zooplankton` class overrides the mortality function defined by the :class:`Plankton` class

::

  class Zooplankton(Plankton):

    # ...

    @veros_method(inline=True)
    def mortality(self, vs):
        """
        Zooplankton mortality is modelled with a quadratic mortality rate
        """
        return self.mortality_rate * self ** 2

By using this approach you only have to focus on the differences between tracers.


Rules
-----

Creating your tracers as objects does not in itself add any time evolution to the system. You must
also specify the interaction between the tracers. This is done by creating rules. A rule specifies
the flow from one tracer to another. An ecosystem can be defined as a collection of rules each
specifying part of the flow between tracers.

Rules consist of a function describing the interaction, the name of the source tracer and the name
of the sink tracer. The function itself may be used in several rules.
The rule function has access to any variable stored in the Veros object. This includes results of
the methods described in the previous section. An example rule could look like

::

  @veros_method(inline=True)
  def recycling_to_po4(vs, source, sink):
    return {source: -vs.recycled[source], sink: vs.redfield_ratio_PN * vs.recycled[source]}

The function returns a dictionary. The keys of the dictionary must be names of the tracers, which
are affected by the rule. The values are numpy arrays corresponding to the change in the tracer.
The return dictionary is not strictly required to contain two keys. If a rule only represents part
of an interaction, just one key can be included. Any number of entries in the dictionary will be
processed, but a rule is intended to represent a flow between two tracers.
The rule should then be registered with the names of the source and sink to make it available for
use in Veros.

::

  register_npzd_rule(vs, 'recycle_phytoplankton_to_po4', (recycling_to_po4, 'phytoplankton', 'po4'))

The rule is registered with the Veros object as the first argument followed by a unique name for the rule
and a tuple consisting of the rule function, the name of the source, and the name of the sink. Those
two names will be passed as arguments to the function. The rule name is used for selecting the rule
for activation.
The tuple may also be replaced by a list containing names of other rules. This collection of rules
may later be activated using just the name the list was registered with.

Optional arguments
++++++++++++++++++

Rules can also be registered with optional arguments.

The :attr:`label` argument specifies a displayed
name which is shown in the graph generated by the biogeochemistry diagnostics.

:attr:`boundary` may take 3 values. 'SURFACE', 'BOTTOM' or None (default). If 'SURFACE' the rule only applies
to the top layer of the grid. 'BOTTOM' means the rule only applies to the cells immediately above the
bottom. None means the rule applies to the entire grid.

:attr:`group` specifies in which of three execution locations the rule will be applied. The 'PRIMARY'
group is the default group. Rules in this group will be evaluated several times in a loop. The number
of times specified by the ratio between :attr:`vs.dt_tracer` and :attr:`vs.dt_bio`. The result of the
rule will be time stepped and added to the tracer concentrations. The 'PRE' group is evaluated once
per tracer time step before the 'PRIMARY' loop. The results of these rules are not time stepped before
adding to the result to the relevant tracers. The 'POST' group is evaluated once before the 'PRIMARY'
rules. Time stepping is left out of 'PRE' and 'POST' rules in order to allow them to clean up or
reuse results from other rules.

Difference between rules and tracer classes
+++++++++++++++++++++++++++++++++++++++++++

The difference between rules and classes and their methods is, that the tracer objects themselves do not
modify tracer concentrations. Only the rules should influence the time evolution of the tracers.
The results of the methods may be used in rules.

Activation
----------

In order to use the created classes and rules. They must be activated. Tracers are activated by register_npzd_data. Rules are activated by adding their names to npzd_selected_rules for example.

::

  detritus = Recyclable_tracer(vs.detritus, 'detritus',
                               sinking_speed=dtr_speed,
                               recycling_rate=vs.remineralization_rate_detritus)
  register_npzd_data(vs, detritus)

This adds a tracer with the name 'detritus' to the model which sinks and a recycling method.

Rules which have been registered with register_npzd_rule are activated by selecting them with
select_npzd_rule. select_npzd_rule accepts rule names. If the name represents a collection of
rules, each rule in the collection is activated.


::

  # activate the npzd_basic_phytoplankton_grazing rule
  select_npzd_rule(vs, 'npzd_basic_phytoplankton_grazing')

  # a list of rules, which have been registered with a single name
  # may be activated collectively from that name

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

  select_npzd_rule(vs, 'group_npzd_basic')  # This activates all the rules in the collection

The example setup file for biogeochemistry demonstrates how a configuration file can be
used to activate rules.
