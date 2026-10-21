# Single-step compatibility verification

The compatibility implementation keeps one compiled numerical timestep and host orchestration for execution and first-order checkpoint replay. Automatic execution selection honors declared native JAX hooks, diagnostics, backend, and solver support. Unsupported configurations retain ordinary execution with a reported reason.

Verification uses float64 and JAX/jaxlib 0.11.2. CPU checks use four logical CPUs of an AMD Ryzen 5 5600. Numerical cases run serially in fresh processes with a process-tree memory limit and a 4 GiB available-system-memory reserve. Checkpoint slot and byte limits cover stored records. The process guard also covers compilation, live states, and differentiation buffers.

| CPU check | Result | Peak process-tree RSS |
| --- | --- | ---: |
| Execution selection, real CLI execution, legacy observers, forcing inputs, checkpoint schedules and storage | 43 passed | 1,753 MiB |
| Streamfunction and pressure forward parity, diagnostic records, split continuation | Passed separately | 1,295 MiB maximum |
| Failure handling, configuration boundaries, output and persistent restarts | Passed separately | 1,584 MiB maximum |
| Native numerical hooks and plugins | Passed | 2,192 MiB |
| Runtime floating settings and plugin parameter derivative | Passed | 1,481 MiB |
| Full-physics trajectory gradient, two finite-difference scales, JVP and split-gradient agreement | Passed | 3,036 MiB |
| Initial temperature/salinity derivatives with EKE, IDEMIX and TKE | Passed | 3,787 MiB |
| Continuous carried controls | Passed | 1,987 MiB |
| Full-physics RAM/disk checkpoint replay against direct trajectory derivatives, including XLA memory accounting | Passed | 3,001 MiB |
| Fixed-mixing checkpoint replay | Passed | 1,905 MiB |
| Runtime parameter and initial-state checkpoint gradients | Passed | 2,501 MiB |
| Nonintegral diagnostic schedules | Passed | 808 MiB |
| Accelerated TDMA and guarded arithmetic derivative tests | Eight passed | 689 MiB maximum |
| NumPy ordinary execution and real CLI execution | Passed | 223 MiB |
| Compilation, synchronized timing and derivative benchmark smoke check | Passed | 1,783 MiB |

The physical cases use small ACC basic grids, generally 8 × 8 × 15. The full-physics trajectory derivative check spans six steps. The checkpoint comparison spans seven steps and includes a terminal loss, differentiable averaging windows, partial accumulation at the final timestep, and both periodic and binomial RAM/disk schedules. Native CPU TDMA is enabled in these checks.

Legacy observer verification compares sampling indices and output times with ordinary execution at a nonintegral output frequency. It writes and restores an unfinished diagnostic accumulation window through a persistent restart, then checks the next output and reset. Forcing replay tests change the source file after the original traversal and verify that recomputation uses the recorded arrays. Host observers run once per original forward step. Missing derivative rules fail during preflight before observers or checkpoint writes.

Distributed verification uses MPI and pure JAX TDMA on one, two, and four CPU ranks. The small halo-coupled recurrence checks shared and rank-local controls, initial-array cotangents, terminal and record losses, RAM/disk replay, recorded forcing, divergent schedule rejection, collective host failures, and cleanup. The physical ocean kernels check both pressure formulations, global diagnostic objectives, direct and replayed gradients, finite differences, halo transpose identities, numerical diagnostics, and extremum derivatives.

| Distributed check | Result | Peak process-tree RSS |
| --- | --- | ---: |
| Halo recurrence, replayed forcing, shared and rank-local gradients, configuration rejection, host failure cleanup, one/two/four ranks | Passed | 2,940 MiB |
| Streamfunction ocean trajectory and checkpoint replay, one/two/four ranks | Passed, including agreement across rank counts | 6,276 MiB |
| Pressure ocean trajectory and checkpoint replay, one/two/four ranks | Passed, including agreement across rank counts | 5,983 MiB |

The distributed numerical runtime uses Python 3.12, mpi4py 4.1.2, mpi4jax 0.9.1.post2, and MPICH. Single-process tests use Python 3.14. Four-rank ocean verification uses a 7 GiB process-tree cap. A first streamfunction attempt stopped at the 6 GiB guard and was completed under the measured larger limit. The system-memory reserve remained 4 GiB.

GPU checks use an NVIDIA RTX 3060 with 12 GiB, native GPU TDMA, disabled XLA preallocation, and a 4 GiB GPU process-memory limit.

| GPU check | Result | Peak host RSS | Peak GPU memory |
| --- | --- | ---: | ---: |
| Execution compatibility, forcing derivatives, TDMA and guarded arithmetic | 13 passed | 1,553 MiB | 278 MiB |
| Streamfunction forward parity, diagnostics and continuation | Passed | 1,594 MiB | 280 MiB |
| Pressure forward parity, diagnostics and continuation | Passed | 1,490 MiB | 260 MiB |
| Full-physics finite differences, JVP and split-gradient agreement | Passed | 4,311 MiB | 458 MiB |
| Full-physics periodic and binomial RAM/disk replay | Passed | 4,130 MiB | 458 MiB |
| Streamfunction ocean trajectory and checkpoint replay, one/two GPU MPI ranks | Passed, including agreement across rank counts | 4,184 MiB | 668 MiB |
| Pressure ocean trajectory and checkpoint replay, one/two GPU MPI ranks | Passed, including agreement across rank counts | 4,050 MiB | 572 MiB |

The GPU forward and derivative tests use 8 × 8 × 15 grids. The seven-step GPU checkpoint comparison uses 6 × 6 × 15. Full-physics GPU derivative verification uses a 5 GiB host-memory cap after the first finite-difference attempt stopped at the 4 GiB guard. Guard stops are incomplete checks. Only the completed reruns count as passes.

GPU MPI checks share the single RTX 3060 between ranks, use pure JAX TDMA, and stage MPI transfers through host memory with `MPI4JAX_USE_CUDA_MPI=0`. They use a 6 GiB process-tree cap and a 4 GiB combined GPU-memory cap. Multiple physical GPUs and direct CUDA-aware MPI transfers are outside this acceptance matrix.

Checkpoint statistics include state, adjoint and parameter bytes, forcing-record peaks, and executable signature counts. `forward_memory` and `reverse_memory` provide XLA estimates for argument, output, alias, temporary and generated-code bytes. These estimates exclude the host compilation cache and external solver allocations, which contribute to the measured process peaks.

## Reproduction

Run the numerical tests individually in fresh processes. Select the JAX backend explicitly, set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`, and disable XLA preallocation. Use the appropriate device and native TDMA setting for the interpreter's installed extensions.

The main test files are `test/execution_test.py`, `test/integration_test.py`, `test/checkpointing_test.py`, `test/dynamic_parameters_test.py`, `test/diagnostics_test.py`, `test/tdma_ad_test.py`, and `test/operators_ad_test.py`. Distributed subprocess tests are in `test/distributed/checkpointing_mpi_test.py` and `test/distributed/integration_mpi_test.py`. Their ocean kernels can also run directly through `mpiexec` to reduce parent-process memory.

## Verification limits

These checks establish agreement for the tested configurations and horizons. The [unstable-column gradient investigation](tke-gradient-investigation.md) records a separate 96-step failure with the default neutral-mixing policy. This compatibility implementation does not change that policy. Gradients for longer trajectories require checks of both derivative modes and finite differences for the intended scientific configuration.

Full 1-degree experiments require more memory than this verification host provides. PETSc combinations are not included in this acceptance matrix because `petsc4py` is unavailable.

Formatting, Flake8, Python compilation, and whitespace checks pass for the changed implementation and tests. The updated tutorial and API pages build with Sphinx's warnings-as-errors setting in a focused documentation project using real NumPy and JAX imports. The repository-wide documentation build also executes the analysis tutorial's external assets and encounters NumPy mocking/JAX import errors in this environment. It is not included as a passing check.
