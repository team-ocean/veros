# Shadowing evaluation for Veros statistical responses

This is an early investigation in workstream 7 of the
[optimization implementation plan](optimization-implementation-plan.md).
Its target is sensitivity of long-time ocean statistics to persistent physical
controls. It does not replace a finite-trajectory derivative with fixed initial
conditions, or resolve the documented neutral-slope conditioning problems.

## Question and candidates

Determine whether shadowing-based estimators provide useful, affordable
statistical derivatives for the intended Veros configurations. Check whether
those configurations actually exhibit chaotic variability before selecting a
chaotic-response method. Stable equilibria and cycles retain their implicit
response methods.

Start with these candidates:

- [NILSS](https://doi.org/10.1016/j.jcp.2017.06.033), a tangent method that limits
  its least-squares calculation to an unstable subspace. Use it as a prototype
  for a small number of control directions and several observables.
- [NILSAS](https://doi.org/10.1016/j.jcp.2019.06.035), an adjoint method suitable
  for a few scalar objectives and many controls. Evaluate it for forcing,
  calibration, closure, and geometry objectives after the tangent prototype.
- Full least-squares shadowing on small test systems as a numerical comparison,
  and ensemble finite differences as an independent statistical check.

Shadowing alone is not automatically the complete derivative of the physical
invariant statistics. [Ni, 2021](https://doi.org/10.1137/20M1388255) identifies an
unstable contribution omitted by nonintrusive shadowing algorithms. Evaluate
that bias explicitly. If it matters, investigate a full linear-response method
with the missing contribution, such as the
[fast adjoint response approach](https://doi.org/10.1137/22M1522383), and compare
its assumptions and implementation cost with ensemble alternatives.

## Evaluation sequence

1. Establish an understood stable system and small chaotic test systems with
   independently resolved statistical responses. Include a case that exposes
   shadowing bias. Lorenz systems can be useful experiments but are not an
   automatic proof of estimator correctness.
2. Measure tangent growth, leading Lyapunov exponents, near-neutral directions,
   correlation times, and sampling uncertainty in representative ocean runs.
   Estimate the unstable-subspace dimension and resulting memory/work cost.
3. Prototype segmented tangent propagation and QR orthogonalization using the
   existing per-step JIT. Add the inhomogeneous parameter tangent and the
   constrained least-squares solve. Segmentation for conditioning is independent
   of block compilation.
4. Specify the differentiated state, including necessary multistep histories,
   and handle fixed inventories and gauges consistently. Account for periodic
   forcing phase and any neutral time direction according to the chosen method.
   Apply the formulation to the actual discrete numerical map.
5. Compare estimates across segment lengths, unstable-subspace sizes, total
   integration lengths, starting states, and forcing phases. Validate against
   long independent forward ensembles at multiple finite-difference scales,
   separating sampling error from systematic bias.
6. Prototype the adjoint variant when its control/objective scaling is useful.
   Reuse the existing VJP and checkpoint infrastructure while adding the
   shadowing constraints. Measure orthogonalization, replay, and least-squares
   costs as well as numerical timestep costs.
7. Evaluate means first, then variance and smooth exceedance observables. Test
   the resulting gradient in a finite optimization update with independently
   evaluated statistical improvement.

## Decision gate

Adopt an estimator only where its derivative agrees with independently resolved
responses, its bias is acceptably bounded for the study, and its cost fits the
declared resource budget. A bounded tangent or adjoint and a small least-squares
residual alone are insufficient.

Report the validated configuration and attractor, sensitivity to numerical
parameters, sampling uncertainty, runtime, and peak memory. When transitions or
bifurcations invalidate a local response description, report that limitation.
A derivative within one attractor does not establish transition probabilities
or the response of an ensemble spanning multiple regimes.

The first deliverable is a comparative feasibility report. It should decide
whether NILSS/NILSAS is adequate, whether the unstable response contribution
must be included, or whether another estimator is preferable before committing
to a production statistical_response implementation.
