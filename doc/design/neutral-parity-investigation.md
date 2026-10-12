# Surface neutral-coefficient sensitivity

Both rejected PETSc performance trajectories reproduce a sensitivity in the surface neutral-slope stencil. Tiny upstream temperature differences switch its taper, producing a large difference in `K_11`. Replaying the neutral kernel with identical inputs gives identical coefficients in ordinary and compiled execution. A one-ULP temperature perturbation independently reproduces the jump.

The affected columns are stably stratified immediately below the surface. These failures therefore differ from the unstable-column condition in the [96-step gradient investigation](tke-gradient-investigation.md), although both involve the same small slope denominator.

## Reproduction and onset

The checks use the optimized CPU PETSc build, native TDMA, float64, default TKE, and 600-second steps. The CPU case compares compiled PETSc with ordinary PETSc on a 16 × 16 × 15 grid over 32° × 32°. The GPU case compares compiled PETSc with ordinary JAX pressure solves on a 128 × 128 × 15 grid over 60° × 84°.

| Case | First coefficient mismatch | Final mismatch | Maximum final temperature error | Maximum final TKE error |
| --- | --- | --- | ---: | ---: |
| CPU, 24 steps | Step 10, `K_11[6, 2, 14]` | One `K_11` element | 1.07e-14 °C | 3.58e-17 |
| GPU, 12 steps | Step 7, `K_11[34, 73, 14]` | Two `K_11` elements | 1.07e-14 °C | 2.90e-17 |

Indices include the two-cell halo. An earlier exploratory GPU trace with fewer history fields first failed at step 5 and also had intermediate `K_22` mismatches at steps 5 and 6. Exact onset varies with the reproduction context. Both traces are retained in the raw results. All 118 other final model variables pass the original comparison, with `rtol=1e-6`, `atol=1e-6`, and finite values required. The trace records the active temperature and salinity at every step. Final comparisons include all time slots.

Adding history outputs leaves all 119 final model variables identical to the uninstrumented compiled run in all four measured cases. This checks that the instrumentation preserves the trajectory being investigated.

## The surface stencil

In [`isoneutral_diffusion_pre`](../../veros/core/isoneutral/isoneutral.py), `dTdz` and `dSdz` are initialized to zero and updated only through the penultimate vertical face. The `kr=1` east/north stencil nevertheless uses their last entries to compute the surface slopes:

```python
slope = -horizontal_density_gradient / (minimum(0, vertical_density_gradient) - 1e-20)
```

At the surface, this denominator is always `-1e-20`. The gradient immediately below does not enter that particular quadrature term. At the first CPU failure it is about `-0.00137`, and at the first GPU failure about `-0.00185`, consistent with stable stratification in the adjacent interior faces.

At CPU step 10, the compiled trajectory has equal neighboring temperatures. The ordinary trajectory differs by `-1.7763568394002505e-15` °C, one float64 ULP. Its horizontal density gradient is approximately `2.36e-21`, giving a slope of `0.236`. With `iso_slopec=0.01` and `iso_dslope=0.005`, the taper changes from `0.9820137900` at zero slope to zero. The surface quadrature then switches from approximately `982` to the steep-slope floor of `500` m²/s.

Both surface quadrature terms contribute to `K_11`. Their combined weight is `2 * dzw[-1] / (4 * dzt[-1]) = 0.7`, so the resulting coefficient change is

```text
0.7 * (1000 * 0.9820137900 - 500) = 337.4096530 m²/s
```

This matches the measured change from `982.0137900` to `644.6041370`. At the first GPU failure in the published probe, one ULP gives a slope of approximately `0.764` and the same coefficient jump. By the final GPU step, the two neighboring faces `[58, 42, 14]` and `[59, 42, 14]` exchange which trajectory has the exactly zero temperature difference.

The coefficient multiplies an almost-zero horizontal tracer gradient in the diagonal flux. That helps explain why the short trajectories retain tiny temperature and TKE differences despite large coefficient differences. It does not establish accuracy or reliable derivatives over longer horizons.

## Identical-input and boundary-policy controls

The probe replays the first failing step's tracer inputs and fixed diffusivity through the actual neutral kernel. Ordinary and inlined execution give identical `K_11` arrays for each input on both devices. They also reproduce the recorded failing coefficients. Lowering one neighboring temperature by one ULP changes `K_11` by `337.4096530` in an isolated replay, without executing a pressure solve or advancing the model.

An experimental control sets slopes to zero only for the surface quadrature terms. It leaves interior slopes, TKE, and pressure solves active. All 119 final model variables then pass parity in both previously failing configurations. Intermediate recorded coefficients also pass. The standalone probe exposes this control through `--zero-surface-slopes`.

This is evidence for the surface stencil as the amplification mechanism. The existing callback/build controls also reproduced the CPU failure with the original callback and optimized PETSc, while the original build passed. Together, these findings support tiny upstream rounding differences triggering a poorly conditioned coefficient calculation, rather than an incorrect neutral-kernel evaluation. They do not isolate every upstream operation that creates a one-ULP tracer difference.

The experimental policy raises the surface horizontal coefficient to its zero-slope taper in cells that previously reached the steep-slope floor. Adopting it requires a decision about the intended surface mixing boundary condition and validation beyond these short trajectories. Default model physics and parity thresholds remain unchanged. This control also does not validate a remedy for the separate unstable-interior gradient failure.

## Reproducing the investigation

Run each configuration in a fresh process with the selected PETSc installation available. The ordinary GPU solver is `scipy_jax` because this PETSc build is CPU-only.

```bash
PYTHONPATH=. VEROS_DEVICE=cpu VEROS_USE_SPECIAL_TDMA=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/neutral_parity_probe.py --output cpu16.json

PYTHONPATH=. VEROS_DEVICE=gpu VEROS_USE_SPECIAL_TDMA=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/neutral_parity_probe.py \
  --size 128 --steps 12 --rescale-grid --eager-solver scipy_jax \
  --output gpu128.json
```

Repeat with `--zero-surface-slopes` for the boundary-policy control. The probe records per-step comparisons, all final fields, identical-input replays, and the isolated perturbation. It intentionally saves scientific parity failures as measurements. Instrumentation or replay inconsistencies raise errors.

The [raw results](../../benchmarks/neutral-parity-investigation.json) include all four runs, source hashes, software/build information, and memory-guard measurements. Heavy runs executed serially with incremental GPU allocation and a host-memory reserve. The largest sampled process-tree RSS was 2620 MiB.
