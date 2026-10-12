# Gradient sensitivity in unstable columns

The 96-step ACC basic gradient failure is tied to the treatment of neutral slopes in unstably stratified columns. The tested TKE trajectory exposes this condition during startup. The evidence does not support near-zero positive TKE as the cause of this failure.

The investigation used a 16 × 16 × 15 ACC basic grid, 600-second momentum and tracer steps, float64, JAX 0.11.2, accelerated CPU TDMA, and step/block checkpointing with eight-step blocks. The objective was the sum of final interior temperature, differentiated with respect to a scale on `t_star`. Numerical failure codes remained zero. These are probes of one configuration, not validation of a replacement mixing scheme.

## Where the sensitivity starts

`isoneutral_diffusion_pre` computes four sets of slopes using

```python
slope = -horizontal_density_gradient / (minimum(0, vertical_density_gradient) - 1e-20)
```

When the vertical density gradient is nonnegative, the denominator is `-1e-20`. A nearly uniform horizontal temperature field then lies within an extremely narrow slope-taper transition. The derivative of the tapered slope at zero horizontal gradient is approximately `9.82e19` with respect to the slope numerator for this setup.

Wet columns first develop negative buoyancy frequency squared at step 35. In a saved unstable face at step 42, increasing one temperature by one float64 increment (`1.776e-15` °C) gives a horizontal density-gradient increment of about `2.277e-21`. Its slope magnitude becomes 0.228 and the taper changes from 0.982 to zero. These slopes enter both neutral and skew tracer mixing and their diagnosed energy transfers.

Separate forward- and reverse-transformed trajectories agree closely through 40 steps. TKE differences grow from about `5e-16` at step 40 to `1.7e-11` at step 48 and `2.0e-7` at step 56. The TKE sign masks remain identical between those trajectories. Their positive TKE values are at least approximately `1.9e-8`. Increasing the minimum mixing length did not resolve the derivative failure. Floating-point differences in the neutral calculation provide a concrete mechanism for the sensitivity without invoking a near-zero TKE square root.

| Steps | Reverse derivative | Forward derivative |
| --- | ---: | ---: |
| 24 | 14.01219708 | 14.01219708 |
| 40 | 23.31432092 | 23.31432092 |
| 48 | 27.90612831 | 27.90846011 |
| 56 | 80.24775215 | 32.45621509 |
| 96 | 96.39477069 | 54.63464750 |

Removing checkpointing, disabling energy-conservation bookkeeping, and replacing the pressure solve with a dense LU solve did not recover reliable derivatives. The pressure matrix's measured condition number was about 165. Simply increasing the slope denominator floor from `1e-20` to `1e-12` also failed. It produced much larger derivatives. A larger denominator alone is therefore not a validated remedy.

## A controlled policy change

An experimental divider sets slopes to zero when the vertical density gradient is nonnegative. It selects safe numerator and denominator operands before dividing, so these cells also have zero slope derivatives. It retains the ordinary taper at zero slope and leaves the rest of the TKE calculation enabled.

| 96-step check | Original policy | Experimental zero-slope policy |
| --- | ---: | ---: |
| Reverse derivative | 96.39477069 | 54.63372828 |
| Forward derivative | 54.63464750 | 54.63372828 |
| Central difference, perturbation `1e-3` | 54.64297262 | 54.63478453 |
| Central difference, perturbation `1e-4` | 55.97720952 | 54.63374557 |

The experimental forward and reverse values agree as well. Its final temperature sum differs from the original reverse-mode primal by about `1.5e-4` out of `34465.8`. This scalar comparison does not establish parity of all fields or longer trajectories.

This intervention strongly identifies the unstable-column neutral slopes as a cause of the observed failure. It changes the mixing policy, including horizontal coefficients whose taper is evaluated at zero slope. Adopting it requires scientific validation and a decision about horizontal mixing and energy transfers in unstable columns. The experiment has not changed the model's default physics. The existing zero-energy square-root derivative convention remains a separate issue.

## Reproducing the checks

Run each variant in a fresh process to limit compilation-cache memory:

```bash
PYTHONPATH=. VEROS_DEVICE=cpu VEROS_USE_SPECIAL_TDMA=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/tke_gradient_probe.py

PYTHONPATH=. VEROS_DEVICE=cpu VEROS_USE_SPECIAL_TDMA=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/tke_gradient_probe.py --zero-unstable-slopes
```

The probe prints synchronized primal values, both derivative modes, finite differences at three scales, and process peak RSS. The `--zero-unstable-slopes` switch applies the experimental divider only inside that process. The [raw investigation measurements](https://github.com/team-ocean/veros/blob/6432c0c85576c6b9dd2254513e19f2816fff4b2c/benchmarks/tke-gradient-investigation.json) retain the horizon checks and unsuccessful interventions alongside the zero-slope result.
