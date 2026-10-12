# Can observations constrain AMOC resilience?

The scientific target is to determine whether observations constrain the
resilience of the Atlantic meridional overturning circulation (AMOC), and which
measurements would distinguish a resilient ocean from one close to a persistent
circulation transition.

The central, falsifiable question is:

> Can two physically plausible ocean models fit the observed ocean equally well,
> yet respond to the same future forcing with fundamentally different AMOC
> outcomes? If so, what observation breaks that ambiguity?

## Scientific motivation

Observations constrain the ocean's state and history, while resilience depends
on its feedbacks. Different combinations of mixing, surface forcing, and
transport could potentially reproduce similar temperature, salinity, and
circulation while responding differently to perturbations. Whether the
available observations eliminate that freedom is the research question.

A study using Veros found a complicated AMOC stability
landscape with multiple coexisting circulation states
([Lohmann et al., 2024](https://doi.org/10.1126/sciadv.adi4253)). Observational
constraints are also being used to narrow future weakening estimates, with
different approaches reaching different conclusions
([Bonan et al., 2025](https://doi.org/10.1038/s41561-025-01709-0),
[Portmann et al., 2026](https://doi.org/10.1126/sciadv.adx4298)).
The proposed contribution is to test how much resilience is identifiable when
uncertain physical processes vary jointly within observational and physical
constraints. The novelty of this proposal remains to be assessed.

## Experiment

1. Construct a family of oceans consistent with observations. Jointly adjust
   initial conditions, historical forcing, and physically constrained mixing.
   Fit hydrography, circulation, heat and freshwater transports, and their
   observed evolution, accounting for observational error and model discrepancy.
2. Search deliberately for divergent futures. Within that acceptable family,
   optimize for the strongest and weakest response to the same specified
   warming/freshening experiment.
3. Identify the mechanism behind the disagreement. Test salt transport
   feedback, deep stratification, Southern Ocean exchange, convection, and other
   candidate processes through controlled model experiments.
4. Find observations that distinguish those mechanisms. Optimize the location,
   depth, variable, and duration of additional measurements. Test their value
   against withheld observations or an independent model.
5. Check whether the conclusion survives changes in resolution and model
   formulation. A finding confined to one closure or one coarse grid has a
   narrower scientific interpretation.

Let C(D) be the set of physically admissible configurations consistent with
observations D. For a specified forcing experiment F, seek the minimum and
maximum of a response R(theta; F) over theta in C(D).

Initially, R can be multidecadal AMOC weakening. More ambitious objectives are
transition thresholds or recovery following a perturbation. A forcing threshold
requires a specified spatial pattern and forcing protocol. It is not a universal
distance to tipping.

Finding a wide feasible response range gives a constructive demonstration of
ambiguity. Finding a narrow range through local optimization alone does not
establish that uncertainty is small. Global search coverage, multiple starting
points, and structural uncertainty need separate treatment. A feasible range
also does not define a probability distribution without a justified probability
model.

## Mapping to the optimization taxonomy

| Scientific task | Contribution | Response regime |
| --- | --- | --- |
| Sensitivity and mechanisms | Explain which feedbacks govern resilience | Trajectory, seasonal, statistics |
| State and forcing estimation | Establish consistency with historical observations | Trajectory |
| Physical parameter calibration | Explore compensating mixing and forcing errors | Seasonal, statistics |
| Optimal perturbations | Find forcing pathways that produce persistent transitions | Trajectory, adjusted branches |
| Learning parameterizations | Test dependence on admissible alternative closures | Seasonal, statistics |
| Identifiability and observation design | Determine what resilience is observable and what to measure next | Seasonal, statistics |
| Geometry and discretization | Test whether inferred resilience survives numerical refinement | Adjusted branches, statistics |

## Results with substantial scientific value

One possible result is that observations tightly constrain a particular
feedback and thereby exclude a substantial class of vulnerable circulation
states. Another is that observations permit sharply different resilience, but
a specific, feasible measurement distinguishes those states.

## Scope for Veros

The first study should target observation-consistent oceans with maximally
different multidecadal responses. This would inform the larger stability
question using the capabilities in the
[optimization matrix](../../benchmarks/ad_demos/MATRIX.md).

The full ambition additionally needs nonlinear branch tracking,
transition/recovery experiments, real-data inference, and validation in models
with atmospheric and sea-ice feedbacks. A 4-degree ocean configuration is useful
for discovery and method development. Claims about Earth's resilience require
that broader evidence.
