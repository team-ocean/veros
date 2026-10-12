# Compiled PETSc performance

Removing per-column Python exception checks from native CPU TDMA is the largest code-level improvement measured here. Compiled PETSc also outperforms the compiled JAX pressure solver on the tested GPU grids. The benchmarks separate pressure solves, callback overhead, whole-model execution, and compilation. Warm measurements use five synchronized samples in fresh processes. The pressure benchmark performs 60 solves per sample. The integration benchmark resets the initial state for each sample and warms the ordinary runner across the complete measured horizon.

The machine has an AMD Ryzen 5 5600, 16 GB of RAM, and an NVIDIA RTX 3060 with 12 GB. Processes share four logical CPUs. All GPU ranks share the same card. Their MPI measurements are not multi-GPU scaling results. OpenBLAS and OpenMP each use one thread. The software is JAX/jaxlib 0.11.2, PETSc/petsc4py 3.26.0, mpi4py 4.1.2, mpi4jax 0.9.1.post2, and MPICH 5.0. GPU allocation is incremental. Heavy runs execute serially under a process-tree memory guard with at least 3 GB of host memory kept available.

## Solver and callback cost

The pressure benchmark uses the ACC basic geometry over a fixed 60° × 84° domain. It checks the primal against the direct callback and checks the global adjoint dot product. Each topology receives slices of an identical global random RHS in the scaling measurements. Native KSP timing uses the same prepared matrix, preconditioner, and normalized RHS. It excludes input/output copies, normalization, independent residual checks, and JAX dispatch. These timings provide a reference for callback overhead.

| Grid and device | Compiled callback (ms/solve) | Native KSP (ms/solve) |
| --- | ---: | ---: |
| 16 × 16, CPU | 0.479 | 0.419 |
| 64 × 64, CPU | 2.902 | 2.849 |
| 128 × 128, CPU | 12.029 | 11.677 |
| 16 × 16, GPU model | 0.550 | 0.420 |
| 64 × 64, GPU model | 2.983 | 2.779 |

These rows use the existing BCGS/GAMG defaults, including a relative tolerance of `1e-14`. At 64–128 cells per side the callback is within approximately 2–7% of the measured native KSP cost. This comparison includes residual validation and GPU transfers. The GPU rows still solve pressure on the CPU. Their native KSP numbers measure that CPU solver.

The callback changes combine global validity and normalization into one native MPI reduction, combine the two residual-check norms through PETSc split reductions, remove redundant solution initialization, and produce the final array without an intermediate scaled copy. A globally zero RHS returns an exact zero solution without entering KSP. A zero local RHS still participates when another rank has a nonzero RHS. The random-RHS measurements show small, noisy effects from these changes. They do not establish a broad throughput improvement by themselves.

## CPU tridiagonal solves

Cython generated a GIL acquisition after every column solve to check for Python exceptions. The four CPU kernel functions now declare `noexcept`. They operate on C pointers and scalars, with C division enabled. The rebuilt extension uses the same `-g -O2` compiler flags as the baseline, isolating this source change.

| CPU grid, 12 steps | Original native TDMA forward (s) | Fixed native TDMA forward (s) | Original value + gradient (s) | Fixed value + gradient (s) |
| --- | ---: | ---: | ---: | ---: |
| 32 × 32 × 15 | 0.217 | 0.0854 | 1.108 | 0.614 |
| 64 × 64 × 15 | 0.822 | 0.352 | not measured | not measured |
| 128 × 128 × 15 | 3.615 | 2.024 | not measured | not measured |

The 32-cell comparison improves the native path by 2.5× forward and 1.8× for value plus gradient. The ordinary runner's 32-cell run falls from 0.400 to 0.248 seconds. The pure JAX TDMA implementation remains competitive, taking 0.0805 seconds forward and 0.640 seconds for value plus gradient in that configuration. The gains depend on the grid and TDMA backend.

The runtime parser now preserves `use_special_tdma=None`, enabling the existing automatic extension selection. Explicit `True` and `False` overrides retain their behavior. At 32 cells on the GPU, native TDMA takes 0.0464 seconds forward and 0.155 seconds for value plus gradient, versus 0.0593 and 0.226 seconds with pure JAX TDMA. All four runs pass the complete variable and derivative checks.

## PETSc build and algorithm choices

Replacing the original `-g -O` PETSc build and reference BLAS/LAPACK with `-O3 -march=native -mtune=native` and OpenBLAS reduced the same 64 × 64 streamfunction benchmark from **3.829 to 2.872 ms/solve**, about 25% less time. Both runs used the original callback, so this comparison isolates the combined build change. It does not separate compiler and BLAS contributions.

A release build can use the following configure options, with the MPI compiler and OpenBLAS paths adjusted for the installation:

```bash
python configure PETSC_ARCH=arch-veros-opt \
  --with-debugging=0 --with-cc=mpicc --with-cxx=0 --with-fc=0 \
  --with-shared-libraries=1 --with-make-np=2 \
  --with-blaslapack-lib='[/path/to/libopenblas.so]' \
  COPTFLAGS='-O3 -march=native -mtune=native'
make PETSC_ARCH=arch-veros-opt all
```

The benchmark build is CPU-only and retained alongside the original build. PETSc's [installation guidance](https://petsc.org/main/install/install/) describes optimization flags, and its [BLAS/LAPACK guidance](https://petsc.org/main/manual/blas-lapack/) covers library selection. `-march=native` targets the build machine's instruction set. Build petsc4py against the selected PETSc installation for normal deployment.

GMRES at the unchanged `1e-14` tolerance reduced the 128 × 128 single-rank streamfunction solve from 12.029 to 11.249 ms and the 64 × 64 pressure-formulation solve from 2.639 to 2.230 ms. SOR smoothing was substantially slower than the existing Jacobi smoother in these tests. A `1e-12` tolerance also reduced solve time in an operator probe, but that probe alone does not validate trajectory accuracy. The implementation retains the existing solver and tolerance defaults. Users can measure GMRES for their configuration with:

```bash
VEROS_PETSC_OPTIONS='-ksp_type gmres' \
  PYTHONPATH=. python benchmarks/pressure_benchmark.py --size 128
```

With an identical global RHS and GMRES at `1e-14`, the 128 × 128 pressure benchmark takes 10.95, 6.98, and 4.73 ms on one, two, and four CPU ranks respectively. Four ranks give about 2.3× speedup on this shared-memory machine. Small whole-model grids show much less MPI benefit. At 32 cells, the fixed native TDMA path takes 0.0854 seconds on one rank and 0.0791 seconds on two ranks for 12 forward steps.

## Whole-model execution and validation

Whole-model measurements use float64, 600-second momentum and tracer steps, default TKE, neutral/skew mixing, and the setup's enabled diagnostics. The table below uses native TDMA. Most MPI measurements use pure JAX TDMA. The two-rank native comparison is identified explicitly. Derivative runs use step and block checkpointing. Every accepted run checks all model variables against the ordinary runner. Derivative runs additionally check central differences at `1e-3` and `1e-4` and forward/reverse agreement. Compilation and first execution are reported separately from warm execution.

| Grid/device, 12 steps | Compiled pressure solver | Forward (s) | Value + gradient (s) | Ordinary runner (s) |
| --- | --- | ---: | ---: | ---: |
| 32 × 32 × 15, CPU | PETSc | 0.0854 | 0.614 | 0.248, PETSc |
| 32 × 32 × 15, GPU | PETSc | 0.0477 | 0.154 | 0.386, JAX |
| 32 × 32 × 15, GPU | JAX | 0.118 | 0.409 | 0.391, JAX |
| 64 × 64 × 15, GPU | PETSc | 0.119 | not measured | 0.503, JAX |
| 64 × 64 × 15, GPU | JAX | 0.259 | not measured | 0.503, JAX |

At 32 cells, PETSc is 2.5× faster forward and 2.7× faster for value plus gradient than the compiled JAX pressure solver on this GPU. The ordinary GPU runner uses JAX, so its comparison with compiled PETSc includes both compilation and a solver change. Pressure-solver defaults remain unchanged. Select PETSc explicitly when evaluating that configuration.

Profiling 60 warm forward steps at 64 cells attributes approximately 10% of CPU wall time after the TDMA fix and 25% of GPU wall time to `KSPSolve`. The CPU 128-cell profile attributes about 8% to KSP. This limits the whole-model gain available from further pressure-solver optimization. It does not establish that the remaining model kernels reach a hardware bandwidth or compute limit.

Forward compilation and first execution take roughly 4–10 seconds for single-rank runs, and value-plus-gradient compilation roughly 25–60 seconds. MPI compilation is serialized by rank to control memory. Warm speedups therefore do not imply faster short, one-off simulations.

Strict parity rejected a 24-step, 16-cell CPU run using the optimized PETSc build and the rebuilt native TDMA extension, and a 12-step, 128-cell GPU PETSc run. They have mismatches in `K_11` at nearly uniform surface cells. The [parity investigation](neutral-parity-investigation.md) reproduces both failures and identifies the surface neutral-slope stencil as the amplification mechanism. Identical-input neutral-kernel replays agree, while a one-ULP temperature perturbation reproduces the coefficient jump. A surface-only zero-slope experiment restores parity in both configurations. Adopting that boundary policy requires further scientific validation. The CPU mismatch also occurs with the original CPU extension and the original callback linked against the optimized build, while the original build and the JAX solver passed. The [gradient investigation](tke-gradient-investigation.md) describes a related sensitivity in unstable interior columns. These trajectories remain rejected measurements, with the original parity tolerances and default mixing physics. The accepted GPU 128-cell JAX run is a forward-only check, and does not validate a PETSc trajectory or a long-horizon gradient.

The [raw results](../../benchmarks/petsc-performance-results.json) retain every warm sample, compilation cost, buffer estimate, guarded process-tree peak, solver option, and accepted validation scope, along with the two scientific parity failures. Operator tests cover primal, JVP, transpose, higher derivatives, small RHSs, exact zeros, failed solves, and boundary sensitivities. Integration regressions cover both pressure formulations, all six diagnostics, CPU output/restarts, CPU MPI on 1/2/4 ranks, and GPU MPI on two ranks.

## Reproduction

Run each configuration in a fresh process with the selected PETSc installation and Veros dependencies available:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/integration_benchmark.py \
  --shape 32 32 15 --rescale-grid --steps 12 \
  --compiled-solver petsc --eager-solver petsc \
  --checkpoint

PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VEROS_PETSC_OPTIONS='-ksp_type gmres' \
  mpiexec -n 4 python benchmarks/pressure_benchmark.py \
  --size 128 --num-proc 2 2 --solves 60 --repeats 5
```

For a GPU model with CPU-only PETSc, use `VEROS_DEVICE=gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and `--eager-solver scipy_jax`. Use `VEROS_USE_SPECIAL_TDMA=1` or `0` to compare native and pure JAX TDMA explicitly. The compiled solver selection remains independent. The pressure harness initializes with the JAX solver automatically. Larger ACC grids use `--rescale-grid` to preserve the physical domain.
